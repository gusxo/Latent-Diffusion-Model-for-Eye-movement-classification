import numpy as np
import tensorflow as tf
import math

#below codes are from ddpm github, see
#https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py

def linear(timestamps, start=0.0001, end = 0.02):
    """
    linear schedule, proposed in original ddpm paper
    """
    return tf.linspace(start, end, timestamps)

def cosine(timestamps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    t = tf.linspace(0.0, timestamps, timestamps+1) / timestamps
    alphas_cumprod = tf.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return tf.clip_by_value(betas, 0, 0.999)

def sigmoid(timestamps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    t = tf.linspace(0.0, timestamps, timestamps+1) / timestamps
    v_start = tf.convert_to_tensor(start / tau)
    v_start = tf.keras.activations.sigmoid(v_start)
    v_end = tf.convert_to_tensor(end / tau)
    v_end = tf.keras.activations.sigmoid(v_end)
    alphas_cumprod = -((t * (end - start) + start) / tau)
    alphas_cumprod = tf.keras.activations.sigmoid(alphas_cumprod) + v_end
    alphas_cumprod = alphas_cumprod / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return tf.clip_by_value(betas, 0, 0.999)