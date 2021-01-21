
from tensorflow import keras as tfkeras
import tensorflow
import sys, os

from os.path import join, isdir
from os import getcwd, mkdir
from glob import glob
import json

from tensorflow.keras.layers import Dense, Activation, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import History, ModelCheckpoint, TensorBoard, LambdaCallback
from tensorflow.keras.initializers import Constant, he_normal


def SimpleModelBN(input_shape, num_output):
    k_init = he_normal()
    b_init = Constant(0.5)
    k_reg = l2(1e-4)

    layers = [
        ## input batch normalization
        BatchNormalization(input_shape=input_shape),
        ## conv block 1
        Conv2D(64, 3, strides=1,
               kernel_initializer=k_init, bias_initializer=b_init,
               kernel_regularizer=k_reg,
               activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2),

        ## conv block 2
        Conv2D(128, 3, strides=1,
               kernel_initializer=k_init, bias_initializer=b_init,
               kernel_regularizer=k_reg,
               activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2),

        ## conv block 3
        Conv2D(256, 3, strides=1,
               kernel_initializer=k_init, bias_initializer=b_init,
               kernel_regularizer=k_reg,
               activation='relu'),
        BatchNormalization(),
        Conv2D(256, 3, strides=1,
               kernel_initializer=k_init, bias_initializer=b_init,
               kernel_regularizer=k_reg,
               activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2),

        ## conv block 4
        Conv2D(512, 3, strides=1,
               kernel_initializer=k_init, bias_initializer=b_init,
               kernel_regularizer=k_reg,
               activation='relu'),
        BatchNormalization(),
        Conv2D(512, 3, strides=1,
               kernel_initializer=k_init, bias_initializer=b_init,
               kernel_regularizer=k_reg,
               activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2),

        ## conv block 5
        Conv2D(512, 3, strides=1,
               kernel_initializer=k_init, bias_initializer=b_init,
               kernel_regularizer=k_reg,
               activation='relu'),
        BatchNormalization(),
        Conv2D(512, 3, strides=1,
               kernel_initializer=k_init, bias_initializer=b_init,
               kernel_regularizer=k_reg,
               activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2),

        ## FC layers
        Flatten(),
        # Dropout(0.5),
        Dense(2048, kernel_regularizer=k_reg, activation='relu'),
        # Dropout(0.5),
        Dense(2048, kernel_regularizer=k_reg, activation='relu'),
        # Dropout(0.5),

        ## output
        Dense(num_output),
        Activation('softmax'),
    ]

    print("No snapshot found, creating model")
    model = tfkeras.models.Sequential(layers)
    return model
