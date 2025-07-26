import tensorflow as tf
import numpy as np
from tensorflow import keras

import numpy as np
from numpy.typing import ArrayLike, NDArray
import typing
import matplotlib.pyplot as plt
import utils

from ddpm_1d import DenoisingDiffusion
from ae import AE

class LDM():
    def __init__(self,
                 autoencoder:AE=None,
                 diffusion:DenoisingDiffusion=None):
        self.ae = autoencoder
        self.diffusion = diffusion

    def _check_shape_compatibility(self):
        try:
            assert self.ae.get_latent_shape() == self.diffusion.model.input_shape[0]
            assert self.ae.decoder.input_shape == self.diffusion.model.output_shape
            assert self.ae.encoder.input_shape == self.ae.decoder.output_shape
        except Exception as e:
            raise Exception("validation of input-output compatibility between autoencoder and diffusion model is failed.")
        return

    def train(
            self,
            x: NDArray, 
            batch_size: int, 
            y: NDArray,
            *,
            gradient_steps: int=None,
            epochs:int=None,
            loss:typing.Literal['l1', 'l2', 'rmse', 'mse', 'mae', 'sdtw']="mse",
            autoencoder_train_steps:int = None,
            autoencoder_train_epochs:int = None,
            verbose:typing.Literal[0,1]=1,
            augmentation=[],
            random_apply=False,
            ema_decay_factor=0.995,
            update_ema_every=10,
            ):
        """
        Perform training the backbone models; including autoencoder and diffusion models.

        parameters
        ----------
        `x` : training data. (n, time-series, features) shape numpy array.

        `batch_size` : batch size for training.

        `y` : class labels for training data. (n) shape numpy array.

        keyword parameters
        ------------------

        #### `gradient_steps` or `epochs`

        - `gradient_steps` : batch iteration count for training. if `epochs` is given, it is ignored.

        - `epochs` : epoch iteration count for training.

        #### `autoencoder_train_steps` or `autoencoder_train_epcohs`

        - `autoencoder_train_steps` : batch iteration count for autoencoder training.

        - `autoencoder_train_epcohs` : epcoh iteration count for autoencoder training.

        - if above two parameters are not given, skip autoencoder training.

        #### optional

        - `loss` : string which determine loss function. default is 'mse'.

        - `augmentation` : callable function's list for apply augmentation for training.

        - `verbose` : 0 or 1, determine show training process.

        - `random_apply` : when using `augmentation`, function is randomly applied for each batch loop.

        - `ema_decay_factor` : EMA factor

        - `update_ema_every` : frequency of EMA apply

        """
        
        self._check_shape_compatibility()
        lossfunc = self.diffusion._get_loss_function(loss)

        if autoencoder_train_steps is not None or autoencoder_train_epochs is not None:
            print("start autoencoder training...")
            ae_loss = self.ae.train(x,
                                    batch_size,
                                    gradient_steps=autoencoder_train_steps,
                                    verbose=verbose,
                                    loss_function=lossfunc,
                                    epochs=autoencoder_train_epochs,
                                    augmentation=augmentation,
                                    random_apply=random_apply,
                                    )
        else:
            ae_loss = None
            print("skip autoencoder training.")

        if epochs is not None:
            assert isinstance(epochs, int) and epochs > 0, "epochs value is allowed only positive integer"
            gradient_steps = epochs * (len(x) // batch_size)
            print(f"LDM.train() : epochs value {epochs} is converted to {gradient_steps} gradient_steps.")

        if gradient_steps is None:
            raise Exception("gradient_steps or epochs parameters is needed")
        
        batch_gen = self.diffusion.batch_generator(x, self.diffusion.timestamps, batch_size, y=y, aug=augmentation, random_apply=random_apply)
        ldm_train_step = self._get_train_step(loss)

        print("start diffusion training...")
        diffuson_loss = self.diffusion._train(
            generator=batch_gen,
            train_step=ldm_train_step,
            gradient_steps=gradient_steps,
            verbose=verbose,
            ema_decay_factor=ema_decay_factor,
            update_ema_every=update_ema_every,
        )

        return {"ae_loss":ae_loss, "ddpm_loss":diffuson_loss}
    
    def _get_train_step(self, loss):
        diffusion_train_step = self.diffusion._get_train_step(self.diffusion._get_loss_function(loss), warpping_tf_function=False)
        @tf.function
        def _train_step(x_0, t, y):
            z_0 = self.ae.to_latent(x_0)
            return diffusion_train_step(z_0, t, y)
        return _train_step
    
    def sample(self,
               batch_size:int,
               sampling_steps:int,
               epochs:int=1, 
               y:typing.Union[None, int, NDArray]=None,
               return_y:bool=False,
               verbose:typing.Literal[0, 1]=1
               ):
        """
        sampling data using ddim sampler.

        parameters
        ----------

        `batch_size`: size of generate data each epochs.

        `sampling_steps`: define how many iteration of reverse & diffusion process of ddim sampling.

        `epochs`: define iteration counts of generation.

        `y`:
            if diffusion model is un-conditional model, allow only `None`.  

            if diffusion model is conditional model, allow `None` or `int` or `numpy array`.
                `None`: sampling as random label.

                `int`: all data is sampling as given label value.

                `numpy array`: length of array is must be same with `batch_size`. each data is sampling as given array's element values.
        
        `return_y`:
            if diffusion model is un-conditional model, allow only `False`.  

            if diffusion model is conditional model, allow `False` or `True`.
                `False`: return only sampled data.

                `True`: return used `y` also, so return value is (x, y).

        return
        ------
        x : sampled data, length is `batch_size` * `epochs`.

        if `return_y` is True, return (x, y), y is same lengths with x.
        """
        
        self._check_shape_compatibility()
        
        x = np.zeros((batch_size * epochs, *self.ae.decoder.output_shape[1:]), dtype=np.float32)
        ry = np.zeros(x.shape[0], dtype=np.int32)

        yy = None
        if y is not None and isinstance(y, int):
            yy = np.full((batch_size,), y, np.int32)
        elif y is not None:
            yy = y

        for i in range(epochs):
            if y is None and self.diffusion.config["num_classes"] is not None:
                yy = tf.random.uniform((batch_size,), 0, self.diffusion.config["num_classes"], dtype=tf.int32)

            ry[i*batch_size : (i+1)*batch_size] = yy
            x[i*batch_size : (i+1)*batch_size] = self.ae.decoder(self.diffusion.ddim_sample(batch_size, sampling_steps, yy, verbose=verbose)).numpy()

        return x if not return_y else (x, ry)
    
    def sample_variation(self,
                        batch_size:int,
                        sampling_steps:int,
                        target_steps:typing.Union[list, tuple],
                        epochs:int=1,
                        y:typing.Union[None, int, NDArray]=None,
                        return_y=False,
                        verbose:typing.Literal[0, 1]=1,
                        mix_targets:bool=False,
                        ):
        """
        parameters
        ----------
        'target_steps': a list, define which middle-step-restoration data is returned.

                        0 value is return perfectly random noise,

                        `sampling_steps` value is return final restoration values.
                        
                        (same as `ldm.sample()` results.)

        other parameters same as `ldm.sample()` functions.
        """
        
        self._check_shape_compatibility()
        
        x = np.zeros((batch_size * epochs * (len(target_steps) if not mix_targets else 1), *self.ae.decoder.output_shape[1:]), dtype=np.float32)
        ry = np.zeros(x.shape[0], dtype=np.int32)

        yy = None
        if y is not None and isinstance(y, int):
            yy = tf.fill((batch_size,), y, np.int32)
        elif y is not None:
            yy = y

        for i in range(epochs):
            if y is None and self.diffusion.config["num_classes"] is not None:
                yy = tf.random.uniform((batch_size,), 0, self.diffusion.config["num_classes"], dtype=tf.int32)

            latent_x_0s = self.diffusion.ddim_sample(batch_size=batch_size,
                                                     sampling_steps=sampling_steps,
                                                     y=yy,
                                                     return_every_steps=True,
                                                     verbose=verbose)
            x_0s = [self.ae.decoder.predict(latent_x_0s[s], verbose=0) for s in target_steps]
            if not mix_targets:
                ry[i*batch_size*len(target_steps) : (i+1)*batch_size*len(target_steps)] = np.concatenate([yy]*len(target_steps))
                x[i*batch_size*len(target_steps) : (i+1)*batch_size*len(target_steps)] = np.concatenate(x_0s, axis=0)
            else:
                ry[i*batch_size : (i+1)*batch_size] = np.array(yy)
                x[i*batch_size : (i+1)*batch_size] = np.mean(np.stack(x_0s), axis=0)
                

        return x if not return_y else (x, ry)
    
    def plot_ddim_sampling(self, steps:int, y:int=None, cmap_name:str="jet_r", save:str=None, figblocksize:int=3, cols=4, remove_tick=False) -> list:
        """
        denoising and visualizing all of z_0 which from during ddim sampling once.

        parameters
        ----------
        `steps` : ddim sampling steps

        `y` : class label

        `cmap_name` : colormap. see matplotlib's colormap.

        `save` : if given string, skip ploting and save as image.

        `figblocksize` : set plot size of each datas.

        `remove_tick` : if given `True`, remove boundary line of each data's image.

        `cols` : determine data count of each rows.
        """
        rows = int(np.ceil((steps+1)/cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * figblocksize, rows * figblocksize), squeeze=False)
        
        latent_x_0s = self.diffusion.ddim_sample(batch_size=1, 
                                                sampling_steps=steps, 
                                                y=y, 
                                                return_every_steps=True)
        x_0s = []
        for i in range(steps+1):
            r = i//cols
            c = i%cols
            x_0 = self.ae.decoder.predict(latent_x_0s[i], verbose=0)
            x_0s.append(x_0)
            utils.draw_gradation(x_0[0,:,0], x_0[0,:,1], axes[r][c], cmap_name, xlim=[-1,1], ylim=[-1,1])
            axes[r][c].set_title(f"{'noise (start)' if not i else f'{i} steps'}")
            if remove_tick:
                axes[r][c].set_xticks([])
                axes[r][c].set_yticks([])
        
        fig.suptitle("DDIM Sampling")
        fig.tight_layout()
        if save:
            fig.savefig(save)
            fig.close()
        else:
            fig.show()
        
        return x_0s 