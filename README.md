# Latent Diffusion Model for Eye movement classification

implementation of latent diffusion models for 'Electrooculography signal generation with conditional diffusion models for eye movement classification'</br>[paper link](https://doi.org/10.1016/j.bspc.2025.108211)

## 1. Requirements
- tensorflow 2.13 (not supported over 2.16)
- dockerfile to be updated (later...)

## 2. Data
In this paper, two public dataset was used.
- Katakana dataset : [paper link](https://doi.org/10.1371/journal.pone.0192684)
- Arabic number dataset : [paper link](https://doi.org/10.1186/s12984-017-0303-5)


Arabic number dataset requires manual preprocessing before using this script.<br>(preprocess script to be updated)

## 3. Training Model
- sample script file to be updated
```
from ddpm_1d import DenoisingDiffusion
from ae import AE
from ldm import LDM
import numpy as np

# x is time-series data, shape is (n, 256, 2)
# y is data's ground truth (zero-based integer, class label)

ldm = LDM()
ldm.ae = AE(x.shape[1:], [64, 128, 256, 512], [0,1,0,1])
ldm.diffusion = DenoisingDiffusion(ldm.ae.get_latent_shape()[1:], num_classes=int(np.max(y))+1, unit_mults=[1, 2, 4])

log = ldm.train(x, 32, epochs=5000, y=y, autoencoder_train_epochs=1500, loss="mse", verbose=1)
```

## 4. Generate Signal
- sample script file to be updated
```
gen_count = 5
sampling_steps = 10

# class is random
x, y = ldm.sample(gen_count, sampling_steps, return_y = True) 

# class is [1, 2, 3, 4, 5]
x = ldm.sample(gen_count, sampling_steps, y=[1, 2, 3, 4, 5]) 

# class is only 0
x = ldm.sample(gen_count, sampling_steps, y=0) 

# Function that returns the sampling intermediate steps mentioned in the paper as the generation result
x, y = ldm.sample_variation(gen_count, sampling_steps, [5, 8], return_y = True) 

# visualization utilities #

# 1. 1d & 2d style visualize
utils.draw_timeseries_writingdata(x, xlim=[-1,1], ylim=[-1,1]) 
# 2. 2d image style, visualize as table
utils.plot_data(x, cols=gen_count, remove_tick = True) 
```