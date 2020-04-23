import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers

class Simple_MLP:
    def __init__(self):
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10)
        ])

        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    def fit_(self, data, labels, n_epochs=10):
        self.model.fit(data, labels, epochs=n_epochs)

    def eval_(self, test_data, test_labels):
        test_loss, test_acc = self.model.evaluate(test_data, test_labels, verbose=2)
        return test_loss, test_acc


# Script (Work)
MNIST = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = MNIST.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0
#class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

simple_mlp = Simple_MLP()
simple_mlp.fit_(train_images, train_labels, n_epochs=10)
_, test_acc = simple_mlp.eval_(test_images, test_labels)
print("\nTest Accuracy: " + str(test_acc))

