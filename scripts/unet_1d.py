import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, activations
from tensorflow.keras.models import Sequential
import math
import typing
# sinusoidal positional embeds
class SinusoidalPosEmb(keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.config = {"dim":dim, **kwargs}
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.cast(self.emb, dtype=tf.float32)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, t):
        t = tf.cast(t, dtype=tf.float32)
        emb = t[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb
    
    def get_config(self):
        return self.config

class LayerNorm(keras.layers.Layer):
    def __init__(self, input_channels, **kwargs):
        super().__init__(**kwargs)
        self.config = {"input_channels":input_channels, **kwargs}
        initializer = tf.ones_initializer()
        self.g = tf.Variable(initial_value=initializer(shape=(1, 1, input_channels)))

    def call(self, x):
        eps = tf.cast(1e-5, tf.float32)
        mean, var = tf.nn.moments(x, axes=[-1], keepdims=True)
        return (x - mean) * tf.math.rsqrt(var + eps) * self.g
    
    def get_config(self):
        return self.config
    
class Block(keras.layers.Layer):
    def __init__(self, units, groups=8, **kwargs):
        super().__init__(**kwargs)
        self.config = {"units":units, "groups":groups, **kwargs}
        self.proj = keras.layers.Conv1D(units, 3, padding="same")
        self.norm = keras.layers.GroupNormalization(groups=groups)
        self.activation = keras.activations.swish

    def call(self, x, scale, shift):
        x = self.proj(x)
        x = self.norm(x)
        if scale is not None and shift is not None:
            x = x * (scale + 1) + shift
        x = self.activation(x)
        return x
    
    def get_config(self):
        return self.config
    
class Identity(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = {**kwargs}
    def call(self, x):
        return x
    def get_config(self):
        return self.config

class ResnetBlock(keras.layers.Layer):
    def __init__(self, units, xiu, tiu, groups=8, **kwargs):
        super().__init__(**kwargs)
        self.config = {"units":units, "xiu":xiu, "tiu":tiu, "groups":groups, **kwargs}
        ti = keras.Input(tiu)
        tx = keras.activations.swish(ti)
        tx = keras.layers.Dense(units * 2)(tx)
        self.mlp = keras.Model(ti, tx)
        self.block1 = Block(units, groups=groups)
        self.block2 = Block(units, groups=groups)
        self.units = units
        self.res_conv = keras.layers.Conv1D(units, 1) if xiu != units else Identity()

    def call(self, x, t=None):
        scale = shift = None
        if t is not None:
            t = self.mlp(t)[:, None, :]
            scale = t[:, :, :self.units]
            shift = t[:, :, self.units:]
        h = self.block1(x, scale, shift)
        h = self.block2(h, None, None)
        return h + self.res_conv(x)
    
    def get_config(self):
        return self.config
    
def Unet(input_shape, units, unit_mults=(1,2,4,8), resnet_block_groups=8, *, num_classes=None):
    #time embeddings
    time_emb_dim = units * 4
    time_emb_input = keras.Input((), name="time_input")
    time_emb_t = SinusoidalPosEmb(units, name="time_SinusoidalPosEmb")(time_emb_input)
    time_emb_t = keras.layers.Dense(time_emb_dim, name="time_dense_1")(time_emb_t)
    time_emb_t = keras.activations.gelu(time_emb_t)
    time_emb_t = keras.layers.Dense(time_emb_dim, name="time_dense_2")(time_emb_t)

    #(optional) condition embedding
    if num_classes:
        class_i = keras.Input((), name="class_input")
        class_emb = tf.keras.layers.Embedding(num_classes, time_emb_dim, name="class_emb")(class_i)
        time_emb_t = tf.keras.layers.Add(name="class_emb_add")([time_emb_t, class_emb])
    
    #main unet parts
    xi = keras.Input(input_shape, name="x_input")
    x_init = keras.layers.Conv1D(units, 7, padding="same", name="x_init_conv")(xi)
    x_ckpts = []
    x = x_init

    for i in range(len(unit_mults)):
        input_units = units if i == 0 else units * unit_mults[i-1]
        output_units = units * unit_mults[i]

        x = ResnetBlock(input_units, input_units, time_emb_dim, resnet_block_groups, name=f"x_Down_{i+1}_ResnetBlock_1")(x, time_emb_t)
        x_ckpts += [x]

        x = ResnetBlock(input_units, input_units, time_emb_dim, resnet_block_groups, name=f"x_Down_{i+1}_ResnetBlock_2")(x, time_emb_t)

        residual = x
        x = LayerNorm(input_units, name=f"x_Down_{i+1}_LayerNorm")(x)
        x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=input_units, value_dim=input_units, use_bias=False, name=f"x_Down_{i+1}_Attention")(x, x)
        x = keras.layers.Add(name=f"x_Down_{i+1}_Residual_Add")([x, residual])
        x_ckpts += [x]

        if i != len(unit_mults) - 1:
            x = keras.layers.Conv1D(output_units, 3, 2, "same", name=f"x_Down_{i+1}_Downsample")(x)
        else:
            x = keras.layers.Conv1D(output_units, 3, padding="same", name=f"x_Down_FinalConv")(x)

    mid_units = units * unit_mults[-1]
    x = ResnetBlock(mid_units, mid_units, time_emb_dim, resnet_block_groups, name=f"x_Middle_ResnetBlock_1")(x, time_emb_t)

    residual = x
    x = LayerNorm(mid_units, name=f"x_Middle_LayerNorm")(x)
    x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=mid_units, value_dim=mid_units, use_bias=False, name=f"x_Middle_Attention")(x, x)
    x = keras.layers.Add(name=f"x_Middle_Residual_Add")([x, residual])

    x = ResnetBlock(mid_units, mid_units, time_emb_dim, resnet_block_groups, name=f"x_Middle_ResnetBlock_2")(x, time_emb_t)

    for i in range(len(unit_mults)):
        final_out_units = units if i == len(unit_mults) - 1 else units * unit_mults[len(unit_mults) - 2 - i]
        mid_out_units = units * unit_mults[len(unit_mults) - 1 - i]
        concated_input_units = mid_out_units + final_out_units

        x = keras.layers.Concatenate(axis=-1, name=f"x_Up_{i+1}_Concatenate_1")([x, x_ckpts.pop()])
        x = ResnetBlock(mid_out_units, concated_input_units, time_emb_dim, resnet_block_groups, name=f"x_Up_{i+1}_ResnetBlock_1")(x, time_emb_t)

        x = keras.layers.Concatenate(axis=-1, name=f"x_Up_{i+1}_Concatenate_2")([x, x_ckpts.pop()])
        x = ResnetBlock(mid_out_units, concated_input_units, time_emb_dim, resnet_block_groups, name=f"x_Up_{i+1}_ResnetBlock_2")(x, time_emb_t)

        residual = x
        x = LayerNorm(mid_out_units, name=f"x_Up_{i+1}_LayerNorm")(x)
        x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=mid_out_units, value_dim=mid_out_units, use_bias=False, name=f"x_Up_{i+1}_Attention")(x, x)
        x = keras.layers.Add(name=f"x_Up_{i+1}_Residual_Add")([x, residual])

        if i != len(unit_mults) - 1:
            x = keras.layers.UpSampling1D(size=2, name=f"x_Up_{i+1}_Upsample")(x)
        x = keras.layers.Conv1D(final_out_units, 3, padding="same", name=f"x_Up_{i+1}_FinalConv")(x)

    x = keras.layers.Concatenate(axis=-1, name="x_final_Concatenate")([x, x_init])
    x = ResnetBlock(units, units*2, time_emb_dim, resnet_block_groups, name="x_final_ResnetBlock")(x, time_emb_t)
    x = keras.layers.Conv1D(input_shape[-1], 1, name="x_final_out_conv")(x)

    model = keras.Model([xi, time_emb_input], x, name="Unet") if not num_classes else keras.Model([xi, time_emb_input, class_i], x, name="Unet")
    return model