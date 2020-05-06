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

PRUNING_PERCENTAGES = [0.0, 0.10, 0.10, 0.05]

OPTIMIZER_FC = tf.keras.optimizers.Adam(learning_rate=1.2e-3)

MNIST_DATA = keras.datasets.mnist

FASHION_MNIST_DATA = keras.datasets.fashion_mnist