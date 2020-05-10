import tensorflow as tf
from tensorflow import keras

"""
    Hyperparameters, data and constants for the FC network (LeNet-5)
"""

LAYERS = {
    'layer1' : (300, 'relu'),
    'layer2' : (100, 'relu'),
    'layer3' : (10, None)
}

PRUNING_PERCENTAGES = [0.0, 0.03, 0.03, 0.06]





SETTINGS = {
    'split' : 0.1,
    'use_random_init' : False,
    'n_epochs' : 20,
    'eval_test' : True,
    'trials' : 5,
    'prune_iterations' : 10
}

LENET_PRUNE_FRACTIONS = [
    [0.0, 0.513, 0.513, 0.2565],
    [0.0, 0.211, 0.211, 0.1055],
    [0.0, 0.07, 0.07, 0.035],
    [0.0, 0.036, 0.036, 0.018],
    [0.0, 0.019, 0.019, 0.0095],
    [0.0, 1.0, 1.0, 0.5]
]

OPTIMIZER_FC = tf.keras.optimizers.Adam(learning_rate=1.2e-3)

MNIST_DATA = keras.datasets.mnist

FASHION_MNIST_DATA = keras.datasets.fashion_mnist
