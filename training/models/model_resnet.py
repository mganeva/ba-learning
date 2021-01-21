

from tensorflow import keras as tfkeras
import tensorflow
from batch_generator import BatchGenerator
from os.path import join, isdir
from os import getcwd, mkdir
from glob import glob
import sys
import os
import numpy as np

from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, Input, Lambda, BatchNormalization, Add, Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import History, ModelCheckpoint, TensorBoard, Callback
from tensorflow.keras.initializers import RandomNormal, Constant, RandomUniform, he_normal

from lr import CosineAnnealer

max_epochs = 100
iterations_per_epoch = 200
degree = 1
base_lr = 0.01

def polynomial_rate(epoch):
    return base_lr*(1.0 - epoch/max_epochs)**degree

class Normalize(Layer):

    def __init__(self, **kwargs):
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Normalize, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return x / tfkeras.backend.sum(x, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return input_shape


class ResNetModel(object):
    def bn_activation(self, activation):
        #act = Lambda(lambda x: tfkeras.activations.get(activation)(x))
        act = Activation(activation)
         
        if self.batch_norm:
            bn = BatchNormalization(gamma_regularizer=self.k_reg, beta_initializer=Constant(0.5), momentum=0.95, epsilon=1e-5)
            #return Lambda(lambda x: act(bn(x)))
            return lambda x: act(bn(x))
        else:
            return act
            
    def conv_layer(self, channels, ksize, activation, strides=1):
        layer = Conv2D(
            channels, ksize,
            kernel_initializer=self.k_init,
            bias_initializer=self.b_init,
            kernel_regularizer=self.k_reg,
            padding="same",
            strides=1,
            use_bias=not self.batch_norm
        )
        return lambda x: self.bn_activation(activation)(layer(x))

    def dense(self, size, activation="relu"):
        layer = Dense(size,
            kernel_regularizer=self.k_reg,
            kernel_initializer=self.k_init,
            bias_initializer=self.b_init,
            use_bias=not self.batch_norm          
        )
        return lambda x: self.bn_activation(activation)(layer(x))
        
    
    def skip_connection(self, channels, ksize, y):
        conv = self.conv_layer(channels, ksize, activation="relu")(y)
        conv2 = self.conv_layer(channels, ksize, activation="relu")(conv)
        return Add()([y, conv2])
                
    """ See https://arxiv.org/abs/1512.03385 """
    def __init__(self, noutput, use_snapshot=True):
        self.callbacks = [
            ModelCheckpoint("snapshots/model.{epoch:04d}.hdf5", monitor='val_loss', save_best_only=True),
            #LearningRateScheduler(polynomial_rate, verbose=1),
            CosineAnnealer(max_epochs, iterations_per_epoch, base_lr),
            TensorBoard(log_dir="./logs", histogram_freq=0, write_grads=False, write_images=False)
        ]

        self.batch_norm = True
        
        #k_init = RandomUniform()
        self.k_init = he_normal()
        self.b_init = Constant(0.5)
        self.k_reg = l2(2.5e-4)

        x = Input(shape=(224, 224, 1))
        y = x
        print(y)

        y = self.conv_layer(64, 7, activation="linear", strides=2)(y)

        y = self.skip_connection(64, 3, y)
        y = self.skip_connection(64, 3, y)
        y = self.skip_connection(64, 3, y)
        y = MaxPooling2D(2)(y)
        print(y)
        
        y = self.skip_connection(64, 3, y)
        y = self.skip_connection(64, 3, y)
        y = self.skip_connection(64, 3, y)
        y = MaxPooling2D(2)(y)
        
        y = self.conv_layer(128, 1, activation="linear")(y)
        y = self.skip_connection(128, 3, y)
        y = self.skip_connection(128, 3, y)
        y = self.skip_connection(128, 3, y)
        y = MaxPooling2D(2)(y)
        
        y = self.skip_connection(128, 3, y)
        y = self.skip_connection(128, 3, y)
        y = self.skip_connection(128, 3, y)
        y = MaxPooling2D(2)(y)
        
        y = self.conv_layer(256, 1, activation="linear")(y)
        y = self.skip_connection(256, 3, y)
        y = self.skip_connection(256, 3, y)
        y = self.skip_connection(256, 3, y)
        
        y = Flatten()(y)
        y = self.dense(2048, activation="relu")(y)
        y = self.dense(2048, activation="relu")(y)
        #y = self.dense(noutput, activation="softplus")(y)
        y = self.dense(noutput, activation="softmax")(y)

        
        # todo: used a named function, since python cannot pickle lambdas
        #y = Normalize()(y)
        print(y)


        snapshots_dir =  join(getcwd(), 'snapshots')

        if not isdir(snapshots_dir):
            mkdir(snapshots_dir)

        self.snapshots = sorted(glob(snapshots_dir + "/model.*.hdf5"))
        self.use_snapshot = use_snapshot

        self.initial_epoch = 0

        self.model = tfkeras.models.Model(x, y)
        #self.optimizer = tfkeras.optimizers.SGD(lr=0.01)#, clipnorm=100.0)
        #self.optimizer = tfkeras.optimizers.Adam(lr=0.0005)#, clipnorm=100.0)
        self.optimizer = tfkeras.optimizers.SGD(clipnorm=10.0, momentum=0.9)

        self.model.compile(
            optimizer=self.optimizer,
            loss='kullback_leibler_divergence',
            metrics=['kullback_leibler_divergence'])
        
        if self.use_snapshot and len(self.snapshots) > 0:
            print("Loading model from snapshot '%s'" % self.snapshots[-1])
            self.model.load_weights(self.snapshots[-1])
            self.initial_epoch = int(self.snapshots[-1].split(".")[-2])
            print("Last epoch was %d" % self.initial_epoch)
            opt = self.model.optimizer
            #print(opt.get_config())
            #print(opt.get_weights())
        else:
            print("No snapshot found")

        print("Layer output shapes:")
        for layer in self.model.layers:
            print(layer.output_shape)

        print("%0.2E parameters" % self.model.count_params())


    def fit(self, train, val, steps_per_epoch=None):
        val_x, val_y = val.load_batches(128)
        return self.model.fit_generator(
            train, epochs=max_epochs, 
            #validation_data=(train.x_test, train.y_test),
            #validation_data=val,
            validation_data=(val_x, val_y),
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

    model = ResNetModel(120)
    train = BatchGenerator(train_path, 8, mean, std)
    val = BatchGenerator(val_path, 8, mean, std)

    model.fit(train, val, steps_per_epoch=iterations_per_epoch)




if __name__ == '__main__':
    run_test()

