from tensorflow.keras.applications import densenet

from training.models.model_simple_BN import SimpleModelBN

def DenseNet121(input_shape, num_output, **kwargs):
    return densenet.DenseNet121(input_shape=input_shape, classes=num_output, weights=None, **kwargs)

def DenseNet169(input_shape, num_output, **kwargs):
    return densenet.DenseNet169(input_shape=input_shape, classes=num_output, weights=None, **kwargs)