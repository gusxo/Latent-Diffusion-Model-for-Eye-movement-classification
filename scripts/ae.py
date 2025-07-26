import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers, activations
from keras.models import Sequential
import math
import types
import pickle
from typing import Literal
import tqdm

from augmentation import batch_generator

class AE():
    def __init__(self, 
                 shape,
                 hidden_units, 
                 downsampling_mask,
                 learning_rate = 0.001):
        self.encoder = self._make_encoder(shape, hidden_units, downsampling_mask, shape[-1])
        self.decoder = self._make_decoder(self.get_latent_shape()[1:], hidden_units[::-1], downsampling_mask[::-1], shape)
        self.opt = tf.keras.optimizers.Adam(learning_rate)
        self.config = {
            "shape":shape,
            "hidden_units":hidden_units,
            "downsampling_mask":downsampling_mask,
            "learning_rate":learning_rate
            }
        """
        DO NOT EDIT.

        it contain used init values when create this class.
        
        it using to save/load model.
        """

    @staticmethod 
    def _make_encoder(input_shape, hidden_units, downsampling_mask, latent_dim):
        xi = keras.Input(input_shape)
        if len(input_shape) == 2:
            conv = layers.Conv1D
        elif len(input_shape) == 3:
            conv = layers.Conv2D
        else:
            raise NotImplementedError()
        
        x = xi
        for units, is_downsample in zip(hidden_units, downsampling_mask):
            x = conv(units, 3, 2 if is_downsample else 1, "same")(x)
            x = layers.Activation(activations.swish)(x)
            
        x = layers.Dense(latent_dim)(x)
        return tf.keras.Model(xi, x, name="encoder")
    
    @staticmethod
    def _make_decoder(input_shape, hidden_units, upsampling_mask, original_shape):
        xi = keras.Input(input_shape)
        if len(original_shape) == 2:
            conv = layers.Conv1DTranspose
        elif len(original_shape) == 3:
            conv = layers.Conv2DTranspose
        else:
            raise NotImplementedError()
        
        x = xi
        for units, is_upsample in zip(hidden_units, upsampling_mask):
            x = conv(units, 3, 2 if is_upsample else 1, "same")(x)
            x = layers.Activation(activations.swish)(x)

        x = layers.Dense(original_shape[-1])(x)
        return tf.keras.Model(xi, x, name="decoder")

    def _get_train_step(self, loss_function=None):
        if loss_function is None:
            loss_func = keras.losses.MeanSquaredError()
        else:
            loss_func = loss_function
        @tf.function
        def train_step(x):
            with tf.GradientTape() as tape:
                z = self.encoder(x)
                x_hat = self.decoder(z)
                loss = loss_func(x, x_hat)
            trainable = self.encoder.trainable_weights + self.decoder.trainable_weights
            gradient = tape.gradient(loss, trainable)
            self.opt.apply_gradients(zip(gradient, trainable))
            return loss
        return train_step

    def train(self, x, batch_size, gradient_steps:int=None, verbose:Literal[0, 1]=1, loss_function:callable=None, epochs:int=None, augmentation=[], random_apply=False):
        progress_bar = None
        try:
            if epochs is not None:
                assert isinstance(epochs, int) and epochs > 0, "epochs value is allowed only positive integer"
                gradient_steps = epochs * (len(x) // batch_size)
                print(f"AE.train() : epochs value {epochs} is converted to {gradient_steps} gradient_steps.")

            if gradient_steps is None:
                raise Exception("gradient_steps or epochs parameters is needed")

            train_step = self._get_train_step(loss_function)
            batch = batch_generator(x, batch_size=batch_size, augmentation=augmentation, random_apply=random_apply)
            assert verbose == 0 or verbose == 1, "verbose parameter is only allowed 0 or 1."
            
            if verbose:
                progress_bar = tqdm.tqdm(total=gradient_steps)

            loss_log = np.zeros(gradient_steps, dtype=np.float32)

            for step in range(gradient_steps):
                loss = train_step(*next(batch)).numpy()

                if verbose:
                    progress_bar.update(1)
                    progress_bar.set_postfix({"loss":loss})

                loss_log[step] = loss

        except Exception as e:
            raise e
        finally:
            if progress_bar is not None:
                progress_bar.close()  
        return loss_log
        
    def to_latent(self, x, batch_size=None):
        if batch_size is None:
            return self.encoder(x, training=False)
        else:
            return tf.convert_to_tensor(self.encoder.predict(x, batch_size=batch_size, verbose=0))
    
    def get_latent_shape(self) -> tuple:
        return self.encoder.output_shape
    
    def save(self, path):
        import os
        parent_dir = os.path.dirname(path)    
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)


        conf = {}
        conf["opt"] = self.opt.variables()
        conf["config"] = self.config
        conf["encoder"] = self.encoder.get_weights()
        conf["decoder"] = self.decoder.get_weights()
        with open(path, "wb") as file:
            pickle.dump(conf, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            conf = pickle.load(file)
        ae = AE(**conf["config"])
        ae.opt.build(ae.encoder.trainable_weights + ae.decoder.trainable_weights)
        ae.opt.set_weights(conf["opt"])
        ae.encoder.set_weights(conf["encoder"])
        ae.decoder.set_weights(conf["decoder"])

        return ae
