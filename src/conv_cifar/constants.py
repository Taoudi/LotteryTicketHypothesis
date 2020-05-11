import tensorflow as tf
from tensorflow import keras

"""
    Hyperparameters, data and constants for the convolutional networks (Conv-2, Conv-4)
"""




SETTINGS_CONV2 = {
    'temp' : 0,
    'n_epochs' : 20
}

SETTINGS_CONV4 = {
    'temp' : 0
}


OPTIMIZER_CONV2 = tf.keras.optimizers.Adam(learning_rate=2e-4)

CIFAR10_DATA = keras.datasets.cifar10