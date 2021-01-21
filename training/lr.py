
import tensorflow
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import numpy as np


class CosineAnnealer(Callback):
    def __init__(self, epochs, steps_per_epoch, base_lr):
        super(CosineAnnealer, self).__init__()
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.base_lr = base_lr

    def on_batch_begin(self, batch, logs=None):
        t = batch * np.pi / self.steps_per_epoch
        lr = self.base_lr * 0.5 * (1 + np.cos(t))
        # print(" Setting learning rate to", lr)
        tensorflow.keras.backend.set_value(self.model.optimizer.lr, lr)


class PolyLearnRate(Callback):
    def __init__(self, base_lr, epochs, steps_per_epoch, warmup_steps=1000, deg=1, verbose=False, initial_epoch=0):
        super(PolyLearnRate, self).__init__()
        self.initial_epoch = initial_epoch
        self.base_lr = base_lr
        self.deg = deg
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.warmup_steps = warmup_steps
        self.verbose = verbose
        self.total_steps = self.epochs * self.steps_per_epoch

        self.initial_step = self.steps_per_epoch * self.initial_epoch

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_begin(self, batch, logs=None):
        current_step = self.current_epoch * self.steps_per_epoch + batch
        delta_step = current_step - self.initial_step

        lr = self.base_lr * (1 - current_step / self.total_steps)**self.deg

        if self.verbose:
            print("\n{} {} {} {}".format(
                self.initial_step, current_step, delta_step, lr))

        if delta_step < self.warmup_steps:
            lr = lr * (delta_step/self.warmup_steps)

        if self.verbose:
            print("poly lr {} {} {}".format(
                self.current_epoch, current_step, lr))

        K.set_value(self.model.optimizer.lr, lr)
