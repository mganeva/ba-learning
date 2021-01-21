import numpy as np


class ImageOp:
    def __init__(self):
        pass

    def __call__(self, x, random_state):
        raise NotImplementedError


class BeamStop(ImageOp):
    def __init__(self, width_range, height_range):
        self.width_range = width_range
        self.height_range = height_range

    def __call__(self, x, random_state):
        mid_index = x.shape[0] // 2
        left = mid_index - \
            random_state.randint(self.width_range[0], self.width_range[1]+1)
        right = mid_index + \
            random_state.randint(self.width_range[0], self.width_range[1]+1)
        height = random_state.randint(
            self.height_range[0], self.height_range[1]+1)
        x[-height:, left:right] = 0.0
        return x


class DetectorMask(ImageOp):
    def __init__(self, min_width, max_width):
        self.min_width = min_width
        self.max_width = max_width

    def __call__(self, x, random_state):
        width = random_state.randint(self.min_width, self.max_width+1)
        x_width = x.shape[1]
        pos = random_state.randint(x_width - width)
        x[:, pos:pos+width] = 0.0
        return x


class RandomCrop(ImageOp):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, x, random_state):
        size_x, size_y = x.shape[0], x.shape[1]
        x_extra = size_x - self.shape[0]
        y_extra = size_y - self.shape[1]
        x_start = random_state.randint(x_extra+1)
        y_start = random_state.randint(y_extra+1)
        return x[x_start:x_start+self.shape[0], y_start:y_start+self.shape[1]]


class Scale(ImageOp):
    def __init__(self, log_scale_range):
        self.log_scale_range = log_scale_range

    def __call__(self, x, random_state):
        log_scale = random_state.uniform(
            self.log_scale_range[0], self.log_scale_range[1])
        return x * 10**log_scale


class Poisson(ImageOp):
    def __call__(self, x, random_state):
        return random_state.poisson(x)


class Log(ImageOp):
    def __init__(self, eps):
        self.eps = eps

    def __call__(self, x, random_state):
        return np.log(x+self.eps)


class Reshape(ImageOp):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, x, random_state):
        return x.reshape(self.shape)


class Normalize(ImageOp):
    def __init__(self):
        pass

    def __call__(self, x, random_state):
        return (x-x.mean()) / x.std()
