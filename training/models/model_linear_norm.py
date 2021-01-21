

from tensorflow import keras as tfkeras
import tensorflow
from batch_generator import BatchGenerator
from os.path import join, isdir
from os import getcwd, mkdir
from glob import glob
import sys
import os

from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import History, ModelCheckpoint, TensorBoard
from tensorflow.keras.initializers import RandomNormal, Constant, RandomUniform, he_normal
from tensorflow.keras import backend as K

class ModelLinearNorm(object):
    def __init__(self, noutput, use_snapshot=True):
        k_init = he_normal()
        b_init = Constant(0.5)
        k_reg = l2(1e-4)
        
        self.layers = [
            ## conv block 1
            Conv2D(64, 3, strides=1, input_shape=(224,224,1),
                   kernel_initializer=k_init, bias_initializer=b_init,
                   kernel_regularizer=k_reg,
                   activation='relu'),
            MaxPooling2D(2),

            ## conv block 2
            Conv2D(128, 3, strides=1,
                kernel_initializer=k_init, bias_initializer=b_init,
                kernel_regularizer=k_reg,
                activation='relu'),
            MaxPooling2D(2),

            ## conv block 3
            Conv2D(256, 3, strides=1,
                kernel_initializer=k_init, bias_initializer=b_init,
                kernel_regularizer=k_reg,
                activation='relu'),
            Conv2D(256, 3, strides=1,
                kernel_initializer=k_init, bias_initializer=b_init,
                kernel_regularizer=k_reg,
                activation='relu'),
            MaxPooling2D(2),

            ## conv block 4
            Conv2D(512, 3, strides=1,
                kernel_initializer=k_init, bias_initializer=b_init,
                kernel_regularizer=k_reg,
                activation='relu'),
            Conv2D(512, 3, strides=1,
                kernel_initializer=k_init, bias_initializer=b_init,
                kernel_regularizer=k_reg,
                activation='relu'),
            MaxPooling2D(2),

            ## conv block 5
            Conv2D(512, 3, strides=1,
                kernel_initializer=k_init, bias_initializer=b_init,
                kernel_regularizer=k_reg,
                activation='relu'),
            Conv2D(512, 3, strides=1,
                kernel_initializer=k_init, bias_initializer=b_init,
                kernel_regularizer=k_reg,
                activation='relu'),
            MaxPooling2D(2),

            ## FC layers
            Flatten(),
            # Dropout(0.5),
            Dense(2048, kernel_regularizer=k_reg, activation='relu'),
            # Dropout(0.5),
            Dense(2048, kernel_regularizer=k_reg, activation='relu'),
            # Dropout(0.5),

            ## output
            Dense(noutput),
            Activation('softplus'),
            Lambda(lambda x : x / K.sum(x, axis=-1, keepdims=True)),
        ]

        self.callbacks = [
            ModelCheckpoint("snapshots/model.{epoch:04d}.hdf5", monitor='val_loss'),
            TensorBoard(log_dir="./logs")
        ]

        snapshots_dir =  join(getcwd(), 'snapshots')

        if not isdir(snapshots_dir):
            mkdir(snapshots_dir)

        self.snapshots = sorted(glob(snapshots_dir + "/model.*.hdf5"))
        self.use_snapshot = use_snapshot

        self.initial_epoch = 0

        if self.use_snapshot and len(self.snapshots) > 0:
            print("Loading model from snapshot '%s'" % self.snapshots[-1])
            self.model = tensorflow.keras.models.load_model(self.snapshots[-1])
            self.initial_epoch = int(self.snapshots[-1].split(".")[-2])+1
            print("Last epoch was %d" % self.initial_epoch)
        else:
            print("No snapshot found, creating model")
            self.model = tfkeras.models.Sequential(self.layers)
            self.optimizer = tfkeras.optimizers.Adam(lr=0.0001)

            self.model.compile(
              optimizer=self.optimizer,
              loss='kullback_leibler_divergence')#,

        print("Layer output shapes:")
        for layer in self.model.layers:
            print(layer.output_shape)

        print("%0.2E parameters" % self.model.count_params())

    def fit(self, train, val):
        return self.model.fit_generator(
            train, epochs=150, 
            validation_data=val,
            callbacks=self.callbacks,
            initial_epoch=self.initial_epoch)


def run_test():
    if len(sys.argv) != 2:
        print("Usage: model_linear_norm.py data_directory")
        sys.exit(0)

    data_path = sys.argv[1]

    train_path = join(data_path, "train")
    val_path= join(data_path, "val")

    ### precomputed from training set
    mean = -12.
    std = 3.0

    model = ModelLinearNorm(120)
    train = BatchGenerator(train_path, 64, mean, std)
    val = BatchGenerator(val_path, 64, mean, std)

    model.fit(train, val)


if __name__ == '__main__':
    run_test()