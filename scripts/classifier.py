import sys
import os
import datetime
import tensorflow as tf
from tensorflow.keras import layers, activations, optimizers, losses
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import typing
import pandas as pd
import seaborn
import augmentation
from numpy.typing import NDArray
# from sklearn.metrics import precision_score, recall_score

def simple_CRNN(input_shape, y_cnt):
    xi = layers.Input(input_shape)

    x = layers.Conv1D(64, 3, 1, "same")(xi)
    x = layers.Activation(activations.swish)(x)
    x = layers.Conv1D(64, 3, 2, "same")(x)

    x = layers.Conv1D(128, 3, 1, "same")(x)
    x = layers.Activation(activations.swish)(x)
    x = layers.Conv1D(128, 3, 2, "same")(x)

    x = layers.Dropout(0.2)(x)

    x = layers.GRU(256, return_sequences=True)(x)
    x = layers.GRU(512, return_sequences=False)(x)
    
    x = layers.Dense(256)(x)
    x = layers.Dense(64)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(y_cnt)(x)
    x = layers.Activation(activations.softmax)(x)

    return tf.keras.Model(xi, x, name="classifier")

def resnet18_1d(input_shape, y_cnt, dropout=0):
    def conv1d_bn(x, filters, kernel_size, strides=1, padding='same'):
        x = layers.Conv1D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def residual_block(x, filters, kernel_size, strides=1, downsample=False):
        shortcut = x
        if downsample:
            shortcut = layers.Conv1D(filters, 1, strides=strides, padding='same', use_bias=False)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = conv1d_bn(x, filters, kernel_size, strides)
        x = conv1d_bn(x, filters, kernel_size, 1)
        
        x = layers.add([x, shortcut])
        x = layers.ReLU()(x)
        return x
    
    inputs = tf.keras.Input(shape=input_shape)

    x = conv1d_bn(inputs, 64, 7, strides=2, padding='same')
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    x = residual_block(x, 64, 3)
    x = residual_block(x, 64, 3)

    x = residual_block(x, 128, 3, strides=2, downsample=True)
    x = residual_block(x, 128, 3)

    x = residual_block(x, 256, 3, strides=2, downsample=True)
    x = residual_block(x, 256, 3)

    x = residual_block(x, 512, 3, strides=2, downsample=True)
    x = residual_block(x, 512, 3)
    if dropout > 0.:
        x = layers.Dropout(dropout)(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(y_cnt, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model



def resnet50_1d(input_shape, y_cnt, dropout=0):

    def conv1d_bn(x, filters, kernel_size, strides=1, padding='same'):
        x = layers.Conv1D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def bottleneck_block(x, filters, kernel_size, strides=1, downsample=False):
        shortcut = x
        if downsample:
            shortcut = layers.Conv1D(filters * 4, 1, strides=strides, padding='same', use_bias=False)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Conv1D(filters, 1, strides=strides, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv1D(filters, kernel_size, strides=1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv1D(filters * 4, 1, strides=1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.add([x, shortcut])
        x = layers.ReLU()(x)
        return x

    inputs = tf.keras.Input(shape=input_shape)

    x = conv1d_bn(inputs, 64, 7, strides=2, padding='same')
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    x = bottleneck_block(x, 64, 3, downsample=True)
    x = bottleneck_block(x, 64, 3)
    x = bottleneck_block(x, 64, 3)

    x = bottleneck_block(x, 128, 3, strides=2, downsample=True)
    x = bottleneck_block(x, 128, 3)
    x = bottleneck_block(x, 128, 3)
    x = bottleneck_block(x, 128, 3)

    x = bottleneck_block(x, 256, 3, strides=2, downsample=True)
    x = bottleneck_block(x, 256, 3)
    x = bottleneck_block(x, 256, 3)
    x = bottleneck_block(x, 256, 3)
    x = bottleneck_block(x, 256, 3)
    x = bottleneck_block(x, 256, 3)

    x = bottleneck_block(x, 512, 3, strides=2, downsample=True)
    x = bottleneck_block(x, 512, 3)
    x = bottleneck_block(x, 512, 3)
    if dropout > 0.:
        x = layers.Dropout(dropout)(x)
    x = layers.GlobalAveragePooling1D()(x)

    outputs = layers.Dense(y_cnt, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

def train(
            train_x:NDArray=None,
            train_y:NDArray=None,
            test_x:NDArray=None,
            test_y:NDArray=None,
            epochs:int=None,
            gradient_steps:int=None,
            learning_rate:int=0.0005,
            batch_size:NDArray=32,
            verbose:typing.Union[typing.Literal[0],typing.Literal[1]]=1,
            plot:typing.Union[typing.Literal[0],typing.Literal[1],str]=0,
            test_name:str="",
            val_x:NDArray=None,
            val_y:NDArray=None,
            train_steps_per_epochs:int=None,
            val_steps_per_epochs:int=None,
            train_gen:typing.Generator=None,
            val_gen:typing.Generator=None,
            num_classes:int=None,
            model=None,
            **kwargs
            ):
    
    if not(plot==0 or plot==1 or isinstance(plot, str)):
        raise NotImplementedError()

    assert (train_x is None or train_y is None) != (train_gen is None)
    assert not ((train_x is None or train_y is None) and (train_gen is None)) , f"must be give parameters : `train_x` and `trian_y` or `train_gen"
    assert (test_x is not None and test_y is not None), f"must be give parameters : `test_x` and `test_y`"
    # assert (val_x is None or val_y is None) != (val_gen is None)
    assert not (epochs is None and gradient_steps is None)
    assert not (num_classes is None and train_y is None)
    assert not (train_gen is not None and train_steps_per_epochs is None), f"when `train_gen` is given, must be `train_steps_per_epochs` is also given. but `train_steps_per_epochs` is {train_steps_per_epochs}"
    assert not (val_gen is not None and val_steps_per_epochs is None)

    if num_classes is None:
        num_classes = int(np.max(train_y)) + 1
    else:
        num_classes = int(num_classes)

    if model is None:
        model = simple_CRNN(train_x.shape[1:] if train_x is not None else next(train_gen)[0].shape[1:], num_classes)
    opt = optimizers.Adam(learning_rate=learning_rate)
    lossfunc = losses.CategoricalCrossentropy()


    def cal_acc(y, y_hat):
        y_label = tf.argmax(y, axis=1)
        y_hat_label = tf.argmax(y_hat, axis=1)
        acc = tf.reduce_mean(tf.cast(tf.equal(y_label, y_hat_label), tf.float32))
        return acc
    
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y = tf.one_hot(y, num_classes)
            y_hat = model(x)
            loss = lossfunc(y, y_hat)
        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
        acc = cal_acc(y, y_hat)
        return loss, acc
    
    @tf.function
    def val_step(x, y):
        y = tf.one_hot(y, num_classes)
        y_hat = model(x)
        loss = lossfunc(y, y_hat)
        acc = cal_acc(y, y_hat)
        return loss, acc

    if train_gen is None:
        train_gen = augmentation.batch_generator(train_x, train_y.astype(np.int8), batch_size=batch_size)
        train_steps_per_epochs = (train_x.shape[0] // batch_size)

    if val_gen is None and val_x is not None:
        val_gen = augmentation.batch_generator(val_x, val_y.astype(np.int8), batch_size=min(batch_size, val_x.shape[0]))
        val_steps_per_epochs = (val_x.shape[0] // min(batch_size, val_x.shape[0]))
    
    if epochs is not None:
        gradient_steps =  train_steps_per_epochs * epochs
    
    if val_gen is not None:
        log = {
            "loss":[],
            "acc":[],
            "val_loss":[],
            "val_acc":[],
        }
    else:
        log = {
            "loss":[],
            "acc":[],
        }

    loss_tmp = []
    acc_tmp = []

    for i in range(gradient_steps):
        loss, acc = train_step(*next(train_gen))
        loss_tmp.append(loss)
        acc_tmp.append(acc)
        
        if (i+1) % train_steps_per_epochs == 0:
            log["loss"].append(np.mean(loss_tmp))
            log["acc"].append(np.mean(acc_tmp))
            loss_tmp = []
            acc_tmp = []

            if val_gen is not None:
                val_loss_tmp = []
                val_acc_tmp = []
                for _ in range(val_steps_per_epochs):
                    loss, acc = val_step(*next(val_gen))
                    val_loss_tmp.append(loss)
                    val_acc_tmp.append(acc)

                log["val_loss"].append(np.mean(val_loss_tmp))
                log["val_acc"].append(np.mean(val_acc_tmp))
            
            if verbose:
                print(f"{(i+1)//train_steps_per_epochs}/{int(np.ceil(gradient_steps/train_steps_per_epochs))} : {' / '.join([f'{key} = {log[key][-1]:.4f}' for key in log.keys()])}")
        
    pred_y = model.predict(test_x, batch_size=batch_size, verbose=0)
    test_acc = np.mean(np.argmax(pred_y, axis=1) == test_y)
    log["test_acc"] = test_acc

    conf_mat = np.zeros((pred_y.shape[1], pred_y.shape[1]), np.int32)
    for true, pred in zip(map(int, test_y), np.argmax(pred_y, axis=1)):
        conf_mat[true, pred] += 1
    conf_mat_df = pd.DataFrame(conf_mat, range(pred_y.shape[1]), range(pred_y.shape[1]))
    log["test_conf_mat"] = conf_mat

    if plot != 0:
        # print(f"train acc = {log['acc'][-1]} / test acc = {test_acc}")
        
        if val_gen is None:
            fig, axes = plt.subplots(1, 2, figsize=(4 * 2, 4))
            fig.suptitle(test_name)
            plt.subplot(1, 2, 2)
            seaborn.heatmap(conf_mat_df, annot=True)

            axes[0].plot(log["acc"])
            axes[0].plot(log["loss"])
            axes[0].legend(["train_accuracy", "train_loss"])
            axes[0].set_ylim([0,1])
        else:
            fig, axes = plt.subplots(1, 3, figsize=(4 * 3, 4))
            fig.suptitle(test_name)
            plt.subplot(1, 3, 3)
            seaborn.heatmap(conf_mat_df, annot=True)

            axes[0].plot(log["loss"])
            axes[0].plot(log["val_loss"])
            axes[0].legend(["train_loss", "val_loss"])

            axes[1].plot(log["acc"])
            axes[1].plot(log["val_acc"])
            axes[1].legend(["train_acc", "val_acc"])

        fig.tight_layout()

        if isinstance(plot, str):
            plt.savefig(plot)
            plt.close()
        else:
            plt.show()

    return log


