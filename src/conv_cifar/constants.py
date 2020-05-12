import tensorflow as tf
from tensorflow import keras

"""
    Hyperparameters, data and constants for the convolutional networks (Conv-2, Conv-4)
"""




SETTINGS_CONV2 = {
    'temp' : 0,
    'n_epochs' : 1
}

SETTINGS_CONV4 = {
    'temp' : 0,
    'n_epochs' : 10,
    'early_stopping':False,
    'split':0.0,
    'use_random_init':False,
    'eval_test':True
}

SETTINGS_CONV6 = {
    'temp' : 0,
    'n_epochs' : 10,
    'early_stopping':True,
    'split':0.1,
    'use_random_init':False,
    'eval_test':True,
    'patience':2
}
# [Conv2, Dense, Output]
PRUNING_PERCENTAGES = [0.1,0.2,0.1]

OPTIMIZER_CONV2 = tf.keras.optimizers.Adam(learning_rate=2e-4)
OPTIMIZER_CONV4 = tf.keras.optimizers.Adam(learning_rate=3e-4)
OPTIMIZER_CONV6 = tf.keras.optimizers.Adam(learning_rate=3e-4)

CIFAR10_DATA = keras.datasets.cifar10