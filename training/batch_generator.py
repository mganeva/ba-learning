import numpy as np
from os.path import join, split, basename
from glob import glob
from math import ceil, floor
from multiprocessing import Pool, cpu_count
import sys
import os
import signal
import tensorflow.python.keras as keras

class WorkerDone(Exception):
    def __init__(self):
        super(WorkerDone, self).__init__()


def load_data(filename, rs, image_ops=None):
    x, y = filename

    x = np.load(x)
    y = np.load(y)

    image_ops = image_ops or []

    for image_op in image_ops:
        x = image_op(x, rs)

    return x, y, rs

class BatchGenerator(keras.utils.Sequence):
    def __init__(self, data_path, batch_size, image_ops=None, nproc=None, seed=None, max_examples=None):
        self.image_ops = image_ops
        self.rs = np.random.RandomState(seed)

        distr_files = glob(join(data_path, "distr*.npy"))
        names = [basename(d).split(".")[-2] for d in distr_files]
        names.sort()
        if max_examples is not None and len(names) > max_examples:
            names = names[:max_examples]
        x_names = [join(data_path, "data_" + name + ".npy") for name in names]
        y_names = [join(data_path, name + ".npy") for name in names]

        self.filenames = list(zip(x_names, y_names))
        self.rs.shuffle(self.filenames)

        self.steps_per_epoch = int(floor(len(self.filenames) / batch_size))
        print("Steps per epoch: {0}".format(self.steps_per_epoch))
        self.batch_size = batch_size
        self.idx = 0

        # set up worker pool
        seeds = [self.rs.randint(0, 2**32) for _ in range(self.batch_size)]
        self.pool = Pool(nproc)
        self.next_batch = [None]*self.batch_size

        filenames = self.next_filenames()

        for i, (seed, filename) in enumerate(zip(seeds, filenames)):
            rs = np.random.RandomState(seed)
            self.next_batch[i] = self.pool.apply_async(
                load_data, (filename, rs, self.image_ops))

        print("Done creating BatchGenerator")

    def next_filenames(self):
        if self.idx >= self.steps_per_epoch:
            self.idx = 0
            self.rs.shuffle(self.filenames)

        j0 = self.idx * self.batch_size
        j1 = j0 + self.batch_size
        self.idx += 1

        return self.filenames[j0:j1]

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __getitem__(self, index):
        return self.next()

    def next(self):
        filenames = self.next_filenames()
        batch_done = [False for i in range(self.batch_size)]

        xs = [False for i in range(self.batch_size)]
        ys = [False for i in range(self.batch_size)]
        rs = [False for i in range(self.batch_size)]

        while not all(batch_done):
            for i, filename in enumerate(filenames):
                if not self.next_batch[i].ready():
                    continue

                batch_done[i] = True
                x, y, rs = self.next_batch[i].get(999)
                self.next_batch[i] = self.pool.apply_async(
                    load_data, (filename, rs, self.image_ops))
                xs[i] = x
                ys[i] = y

        return np.array(xs), np.array(ys)

    def terminate(self):
        self.pool.terminate()

    def close(self):
        # for item in self.next_batch:
        #    item.get(60)

        self.pool.close()


def run_tests():
    import gc
    import time

    N = 5
    batch_size = 64

    bg = BatchGenerator(
        "/home/jonathan/data/ba/lattice/100k/train", batch_size)
    try:
        start = time.time()
        for i in range(N):
            x, y = bg.next()
            print(i)

        end = time.time()
        delta = end-start

        image_per_sec = N*batch_size / delta
        print("image per sec:", image_per_sec)
    except:
        bg.terminate()
    finally:
        bg.close()


if __name__ == '__main__':
    run_tests()
