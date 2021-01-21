

from tensorflow import keras as tfkeras
import tensorflow
from batch_generator import BatchGenerator
from os.path import join, isdir
from os import getcwd, mkdir
from glob import glob
import sys
import os
import numpy as np

from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import History, ModelCheckpoint, TensorBoard
from tensorflow.keras.initializers import RandomNormal, Constant, RandomUniform, he_normal

from lr import CosineAnnealer

max_epochs = 300
steps_per_epoch = 1000
base_lr = 0.005
SNAPSHOTDIR='snapshots_CA'

class SimpleModel(object):
    def __init__(self, noutput, use_snapshot=True):
        #k_init = RandomUniform()
        k_init = he_normal()
        b_init = Constant(0.5)
        k_reg = l2(1e-4)
        
        self.layers = [
            ## conv block 1
            Conv2D(64, 3, strides=1, input_shape=(224,224,1),
                kernel_initializer=k_init, bias_initializer=b_init,
                kernel_regularizer=k_reg,
                name="conv2d_1_1",
                activation='relu'),
            MaxPooling2D(2),

            ## conv block 2
            Conv2D(128, 3, strides=1,
                kernel_initializer=k_init, bias_initializer=b_init,
                kernel_regularizer=k_reg,
                name="conv2d_2_1",
                activation='relu'),
            MaxPooling2D(2),

            ## conv block 3
            Conv2D(256, 3, strides=1,
                kernel_initializer=k_init, bias_initializer=b_init,
                kernel_regularizer=k_reg,
                name="conv2d_3_1",
                activation='relu'),
            Conv2D(256, 3, strides=1,
                kernel_initializer=k_init, bias_initializer=b_init,
                kernel_regularizer=k_reg,
                name="conv2d_3_2",
                activation='relu'),
            MaxPooling2D(2),

            ## conv block 4
            Conv2D(512, 3, strides=1,
                kernel_initializer=k_init, bias_initializer=b_init,
                kernel_regularizer=k_reg,
                name="conv2d_4_1",
                activation='relu'),
            Conv2D(512, 3, strides=1,
                kernel_initializer=k_init, bias_initializer=b_init,
                kernel_regularizer=k_reg,
                name="conv2d_4_2",
                activation='relu'),
            MaxPooling2D(2),

            ## conv block 5
            Conv2D(512, 3, strides=1,
                kernel_initializer=k_init, bias_initializer=b_init,
                kernel_regularizer=k_reg,
                name="conv2d_5_1",
                activation='relu'),
            Conv2D(512, 3, strides=1,
                kernel_initializer=k_init, bias_initializer=b_init,
                kernel_regularizer=k_reg,
                name="conv2d_5_2",
                activation='relu'),
            MaxPooling2D(2),

            ## FC layers
            Flatten(),
  #          Dropout(0.5),
            Dense(2048, kernel_regularizer=k_reg, activation='relu', name="dense_1"),
 #           Dropout(0.5),
            Dense(2048, kernel_regularizer=k_reg, activation='relu', name="dense_2"),
#            Dropout(0.5),

            ## output
            Dense(noutput, name="output"),
            Activation('softmax'),
        ]

        self.callbacks = [
            ModelCheckpoint(SNAPSHOTDIR + "/model.{epoch:04d}.hdf5", monitor='val_loss'),
            CosineAnnealer(max_epochs, steps_per_epoch, base_lr),
            TensorBoard(log_dir="./logs")#, histogram_freq=1, write_grads=True, write_images=True)
        ]

        snapshots_dir =  join(getcwd(), SNAPSHOTDIR)

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
            opt = self.model.optimizer
            #print(opt.get_config())
            #print(opt.get_weights())
        else:
            print("No snapshot found, creating model")
            self.model = tfkeras.models.Sequential(self.layers)
            self.optimizer = tfkeras.optimizers.SGD()#, clipnorm=100.0)
            #self.optimizer = tfkeras.optimizers.Adam(lr=0.0001)

            self.model.compile(
              optimizer=self.optimizer,
              loss='kullback_leibler_divergence',
              metrics=['kullback_leibler_divergence'])

        print("Layer output shapes:")
        for layer in self.model.layers:
            print(layer.output_shape)

        print("%0.2E parameters" % self.model.count_params())

            

    def fit(self, train, val, max_epochs=None, steps_per_epoch=None):
#        val_x, val_y = val.load_batches(16)
        return self.model.fit_generator(
            train, epochs=max_epochs, 
            #validation_data=(train.x_test, train.y_test),
            validation_data=val,
            #validation_data=(val_x, val_y),
            callbacks=self.callbacks,
            initial_epoch=self.initial_epoch,
            steps_per_epoch=steps_per_epoch)


def run_test():
    # get data directory if specified
    if len(sys.argv) == 2:
        data_path = sys.argv[1]
    else:
        data_path = "./data"
    
    train_path = join(data_path, "train")
    val_path= join(data_path, "val")

    ### precomputed from training set
    mean = -12.
    std = 3.0

    model = SimpleModel(120)
    train = BatchGenerator(train_path, 16, mean, std)
    val = BatchGenerator(val_path, 16, mean, std)

    model.fit(train, val, max_epochs=max_epochs, steps_per_epoch=steps_per_epoch)


if __name__ == '__main__':
    run_test()