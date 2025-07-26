import tensorflow as tf
import numpy as np
from tensorflow import keras
import tqdm

import numpy as np
from numpy.typing import ArrayLike, NDArray
import typing
import copy
import pickle

import augmentation
import beta_scheduler
from unet_1d import Unet

class DenoisingDiffusion():
    def __init__(self, 
                 input_shape : ArrayLike,
                 units : int = 64,
                 unit_mults : ArrayLike = [1,2,4,8],
                 resnet_block_groups : int = 8,
                 num_classes : typing.Union[None, int] = None,
                 schedule : typing.Literal["linear", "cosine", "sigmoid"] = "linear",
                 timestamps : int = 1000,
                 ):
        

        self.config = {
            "input_shape":input_shape,
            "units":units,
            "unit_mults":unit_mults,
            "resnet_block_groups":resnet_block_groups,
            "num_classes":num_classes,
            "schedule":schedule,
            "timestamps":timestamps,
        }
        """
        DO NOT EDIT.

        it contain used init values when create this class.
        
        it using to save/load model.
        """

        self.model = Unet(input_shape,
                          units=units,
                          unit_mults=unit_mults,
                          resnet_block_groups=resnet_block_groups,
                          num_classes=num_classes)
        
        if schedule == "linear":
            self.betas = beta_scheduler.linear(timestamps)
        elif schedule == "cosine":
            self.betas = beta_scheduler.cosine(timestamps)
        elif schedule == "sigmoid":
            self.betas = beta_scheduler.sigmoid(timestamps)
        else:
            raise NotImplementedError()
        
        self.dtype = tf.float32
        self.betas = tf.constant(self.betas, dtype=self.dtype)
        self.alpha_bar = np.cumprod(1 - self.betas)
        self.alpha_bar = tf.constant(self.alpha_bar, dtype=self.dtype)

        #below value is used for ddpm sampler
        self.alpha_bar_prev = np.append(1.0, self.alpha_bar[:-1])
        self.posterior_mean_coef1 = self.betas * np.sqrt(self.alpha_bar_prev) / (1 - self.alpha_bar)
        self.posterior_mean_coef2 = (1 - self.alpha_bar_prev) * np.sqrt(1 - self.betas) / (1 - self.alpha_bar)
        self.posterior_variance = self.betas * (1 - self.alpha_bar_prev) / (1 - self.alpha_bar)
        self.posterior_log_var_cliped = np.log(np.maximum(self.posterior_variance, 1e-20))

        self.weights = self.model.get_weights()
        self.ema_weights = copy.deepcopy(self.weights)
        self.opt = None
        self.timestamps = timestamps

    def save_pickle(self, path):
        import os
        parent_dir = os.path.dirname(path)    
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        pkl = {}
        pkl["config"] = self.config
        pkl["opt_weights"] = self.opt.variables() if self.opt is not None else None
        pkl["opt_conf"] = self.opt.get_config() if self.opt is not None else None
        pkl["weights"] = self.weights
        pkl["ema_weights"] = self.ema_weights
        with open(path, "wb") as file:
            pickle.dump(pkl, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pickle(path, optimizer_class=tf.keras.optimizers.Adam):
        with open(path, "rb") as file:
            pkl = pickle.load(file)
        diffusion = DenoisingDiffusion(**pkl["config"])
        diffusion.weights = pkl["weights"]
        diffusion.ema_weights = pkl["ema_weights"]
        diffusion.model.set_weights(diffusion.weights)
        if pkl["opt_conf"] is not None:
            diffusion.opt = optimizer_class.from_config(pkl["opt_conf"])
            diffusion.opt.build(diffusion.model.trainable_weights)
            diffusion.opt.set_weights(pkl["opt_weights"])
        return diffusion

    def diffusion_process(self, x_start, x_end, t):
        return x_start * tf.sqrt(tf.gather(self.alpha_bar, t)[:, None, None]) + x_end * tf.sqrt(1.0 - tf.gather(self.alpha_bar, t)[:, None, None])
    
    def _get_train_step(self, loss_func, warpping_tf_function=True):
        def _train_step(*args):
            x_0 = args[0]
            t = args[1]

            with tf.GradientTape() as tape:
                noise = tf.random.normal(shape=(x_0.shape), dtype=self.dtype)
                x_t = self.diffusion_process(x_0, noise, t)
                pred_noise = self.model([x_t, *args[1:]], training=True)
                loss = loss_func(noise, pred_noise)
            #apply gradient
            gradient = tape.gradient(loss, self.model.trainable_weights)
            self.opt.apply_gradients(zip(gradient, self.model.trainable_weights))
            return loss
        
        return tf.function(_train_step) if warpping_tf_function else _train_step
    
    def update_ema(self, decay_factor):
        for i in range(len(self.weights)):
            self.ema_weights[i] = decay_factor * self.ema_weights[i] + (1 - decay_factor) * self.weights[i]

    def _get_loss_function(self, str):
        if str=="l1" or str=="mae":
            loss_func = keras.losses.MeanAbsoluteError()
        elif str =="l2" or "mse":
            loss_func = keras.losses.MeanSquaredError()
        elif str =="rmse":
            loss_func = lambda t, p: tf.sqrt(tf.reduce_mean(tf.square(t - p)))
        elif str == "sdtw":
            try:
                from softdtwkeras.SDTWLoss import SDTWLoss
                loss_func = SDTWLoss(gamma=0.5)
            except Exception as e:
                raise Exception("failed to load SDTW lib. see 'https://github.com/AISViz/Soft-DTW' and install it.")
        elif str == "sdtw_mse":
            try:
                from softdtwkeras.SDTWLoss import SDTWLoss
                sdtw_loss = SDTWLoss(gamma=0.5)
                mse_loss = keras.losses.MeanSquaredError()
                loss_func = lambda t, p: tf.reduce_mean([sdtw_loss(t,p), mse_loss(t,p)])
            except Exception as e:
                raise Exception("failed to load SDTW lib. see 'https://github.com/AISViz/Soft-DTW' and install it.")
        else:
            raise NotImplementedError()
        return loss_func

    def train(
            self, 
            x:ArrayLike, 
            batch_size:int, 
            *,
            gradient_steps:int=None, 
            epochs:int=None,
            y:typing.Union[ArrayLike, None] = None,
            ema_decay_factor=0.995,
            loss:typing.Literal["l1", "l2", "rmse", "mse", "mae", "sdtw"]="l1",
            update_ema_every:int=10,
            verbose:typing.Literal[0,1]=1,
            augmentation=[],
            random_apply=False,
            ):
        """
        define training loop.
        """

        if epochs is not None:
            assert isinstance(epochs, int) and epochs > 0, "epochs value is allowed only positive integer"
            gradient_steps = epochs * (len(x) // batch_size)
            print(f"DenoisingDiffusion.train() : epochs value {epochs} is converted to {gradient_steps} gradient_steps.")

        if gradient_steps is None:
            raise Exception("gradient_steps or epochs parameters is needed")

        self._check_y(y)
        
        loss_func = self._get_loss_function(loss)

        batch = self.batch_generator(x, timestamps=self.timestamps, y=y, batch_size=batch_size, aug=augmentation, random_apply=random_apply)
        train_step = self._get_train_step(loss_func)

        return self._train(
            generator=batch,
            train_step=train_step,
            gradient_steps=gradient_steps,
            ema_decay_factor=ema_decay_factor,
            update_ema_every=update_ema_every,
            verbose=verbose
        )
    
    def _train(
        self, 
        *,
        generator:typing.Generator,
        train_step:typing.Callable,
        gradient_steps:int,
        ema_decay_factor:float,
        update_ema_every:int,
        verbose:typing.Literal[0,1]=1,
        ):
        assert verbose == 0 or verbose == 1, "verbose parameter is only allowed 0 or 1."
        progress_bar = None
        if self.opt is None:
            print("DenoisingDiffusion._train() : since DenoisingDiffusion.opt is not defined, automaticaly set to Adam optimizer with learning rate = 0.0002")
            print("if you want change learning rate, manually assign Adam DenoisingDiffusion.opt before calling this function.")
            print("Note : you can assign another optimizer(eg. SGD), but if so, you will should pass your optimizer class to 'optimizer_class' parameter when use 'load_pickle()' function.")
            self.opt = keras.optimizers.Adam(0.0002)

        try:

            self.model.set_weights(self.weights)
            loss_log = np.zeros(gradient_steps, dtype=np.float32)
            if verbose:
                progress_bar = tqdm.tqdm(total=gradient_steps)
            
            for step in range(gradient_steps):
                loss = train_step(*next(generator)).numpy()

                #update ema
                if (step % update_ema_every == 0) or (step == gradient_steps-1):
                    self.weights = self.model.get_weights()
                    self.update_ema(ema_decay_factor)

                if verbose:
                    progress_bar.update(1)
                    progress_bar.set_postfix({"loss":loss})
                
                loss_log[step] = loss
        except Exception as e:
            raise e
        finally:
            if progress_bar is not None:
                progress_bar.close() 
        print(f"training complete")
        return loss_log
    
    def batch_generator(self, x, timestamps, batch_size, y=None, aug=[], random_apply=False):
        if y is None:
            gen = augmentation.batch_generator(x, batch_size=batch_size, augmentation=aug, random_apply=random_apply)
        else:
            gen = augmentation.batch_generator(x, y, batch_size=batch_size, augmentation=aug, random_apply=random_apply)

        while 1:
            batch = next(gen)
            t = tf.random.uniform((batch_size,), 0, timestamps, dtype=tf.int32)
            if y is None:
                yield tf.convert_to_tensor(batch[0], dtype=tf.float32), t
            else:
                yield tf.convert_to_tensor(batch[0], dtype=tf.float32), t, tf.convert_to_tensor(batch[1], dtype=tf.float32)

    def _check_y(self, y):
        if self.config["num_classes"] is not None and y is None:
            raise Exception("Since u-net is a conditional model, conditions(y) are required.")
        elif self.config["num_classes"] is None and y is not None:
            raise Exception("u-net is not conditional model, but received the conditions(y).")
        return

    def ddpm_sample(self, *, batch_size=None, steps=None, x_t=None, y=None):
        if steps is None:
            steps = self.timestamps

        self.model.set_weights(self.ema_weights)

        if x_t is None:
            if batch_size is None:
                raise ValueError("batch_size is required when x_t is None or undefined.")
            x_t = tf.random.normal(shape=(batch_size,) + self.model.input_shape[0][1:])
            x_t = tf.cast(x_t, self.dtype)
        else:
            x_t = tf.convert_to_tensor(x_t, dtype=self.dtype)
            if x_t.shape[1:] != self.model.input_shape[0][1:]:
                raise ValueError(f"x_t shape is expected {self.model.input_shape[0]}, but received {x_t.shape}")
            
        self._check_y(y)

        if y is not None and isinstance(y, int):
            y = np.full((x_t.shape[0],), y, np.int32)


        for t in tqdm.trange(steps - 1, -1, -1, desc="sampling steps"):
            tt = np.full((x_t.shape[0],), t, np.int32)
            if y is None:
                pred_noise = self.model([x_t, tt], training=False)
            else:
                pred_noise = self.model([x_t, tt, y], training=False)
            pred_x_start_from_noise = tf.cast(tf.sqrt(1 / self.alpha_bar[t]), dtype=self.dtype) * x_t - tf.cast(tf.sqrt(1 / self.alpha_bar[t] - 1), dtype=self.dtype) * pred_noise
            pred_x_start_from_noise = tf.clip_by_value(pred_x_start_from_noise, -1.0, 1.0)
            model_mean = self.posterior_mean_coef1[t] * pred_x_start_from_noise + self.posterior_mean_coef2[t] * x_t
        
            noise = tf.random.normal(x_t.shape) if t else 0

            x_t = model_mean + tf.cast(tf.exp(0.5 * self.posterior_log_var_cliped[t]), dtype=self.dtype) * noise 

        self.model.set_weights(self.weights)

        return x_t.numpy()
    
    def ddim_sample(self, 
                    batch_size:int, 
                    sampling_steps:int, 
                    y=None, 
                    return_every_steps:bool=False, 
                    verbose:typing.Literal[0, 1]=1
                    ):
        self.model.set_weights(self.ema_weights)

        x_t = tf.random.normal(shape=(batch_size,) + self.model.input_shape[0][1:])
        x_t = tf.cast(x_t, self.dtype)

        self._check_y(y)

        if y is not None and isinstance(y, int):
            y = tf.fill((x_t.shape[0],), y, np.int32)

        seq = range(self.timestamps-1, -1, -self.timestamps//sampling_steps)
        seq_next = list(seq[1:]) + [-1]

        logs = [x_t]
        steprange = tqdm.trange(sampling_steps) if verbose == 1 else range(sampling_steps)
        assert verbose == 0 or verbose == 1, "verbose parameter is only allowed 0 or 1."

        for steps in steprange:
            i = seq[steps]
            j = seq_next[steps]
            t = tf.ones((batch_size,), dtype=tf.int32) * i
            at = self.alpha_bar[i]
            at_next = self.alpha_bar[j] if j >= 0 else tf.ones(1)

            if y is None:
                e_t = self.model([x_t, t], training=False)
            else:
                e_t = self.model([x_t, t, y], training=False)

            x_0_hat = (x_t - e_t * tf.sqrt(1 - at)) / tf.sqrt(at)
            if return_every_steps:
                logs.append(x_0_hat.numpy())

            c1 = 1 * tf.sqrt((1 - at / at_next) * (1 - at_next) / (1 - at))

            c2 = tf.sqrt((1 - at_next) - c1**2)

            x_t = tf.sqrt(at_next) * x_0_hat + c1 * tf.random.normal(x_t.shape) + c2 * e_t

        self.model.set_weights(self.weights)

        return x_t.numpy() if not return_every_steps else logs
    
