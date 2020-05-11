import tensorflow as tf
from tensorflow import keras

"""
    Hyperparameters, data and constants for the FC network (LeNet-5)
"""


"""
    Fully-connected Network structure
"""
LAYERS = {
    'layer1' : (300, 'relu'),
    'layer2' : (100, 'relu'),
    'layer3' : (10, None)
}

"""
    Pruning fractions for each layer every iteration
"""
PRUNING_PERCENTAGES = [0.0, 0.2, 0.2, 0.1]

"""
    Global settings for the fully-connected network
"""
SETTINGS = {
    'split' : 0.0,
    'use_random_init' : False,
    'n_epochs' : 20,
    'eval_test' : True,
    'trials' : 5,
    'lower_bound' : 0.02,
    'patience' : 10
}

"""
    Gradient Descent Optimizer for training the networks
"""
OPTIMIZER_FC = tf.keras.optimizers.Adam(learning_rate=1.2e-3)

"""
    Datasets for the fully-connected network
"""
MNIST_DATA = keras.datasets.mnist

FASHION_MNIST_DATA = keras.datasets.fashion_mnist
