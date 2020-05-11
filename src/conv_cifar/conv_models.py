import tensorflow as tf
from tensorflow.keras import datasets,layers,models,initializers
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

from constants import OPTIMIZER_CONV2, OPTIMIZER_CONV4

class CONV2_NETWORK:
    def __init__(self, dropout=False):
        self.model = models.Sequential()

        self.model.add(layers.BatchNormalization())
        if dropout:
            self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3),kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.BatchNormalization())
        if dropout:
            self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(256,activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.Dense(256,activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.Dense(10))


        self.model.compile(OPTIMIZER_CONV2,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        #self.es = EarlyStopping(monitor='val_loss', patience=SETTINGS['patience'])
        self.bs = 60

    def fit(self,X,Y,SETTINGS):
        history = self.model.fit(X, Y, epochs=SETTINGS['n_epochs'], batch_size=self.bs)
        return history

    def evaluate_model(self, testX, testY):
        test_loss, test_acc = self.model.evaluate(testX, testY, verbose=2)
        return test_loss, test_acc

    def save_(self,name):
        self.model.save(str(name))



class CONV4_NETWORK:
    def __init__(self, dropout=False):
        self.model = models.Sequential()

        self.model.add(layers.BatchNormalization(input_shape=(32, 32, 3)))
        if dropout:
            self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.BatchNormalization())
        if dropout:
            self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.BatchNormalization())
        if dropout:
            self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.BatchNormalization())
        if dropout:
            self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.MaxPooling2D((2, 2)))


        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(256,activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.Dense(256,activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.Dense(10))

        self.model.compile(OPTIMIZER_CONV4,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        #self.es = EarlyStopping(monitor='val_loss', patience=SETTINGS['patience'])
        self.bs = 60

    def fit(self,X,Y,SETTINGS):
        history = self.model.fit(X, Y, epochs=SETTINGS['n_epochs'], batch_size=self.bs)
        return history

    def evaluate_model(self, testX, testY):
        test_loss, test_acc = self.model.evaluate(testX, testY, verbose=2)
        return test_loss, test_acc

    def save_(self,name):
        self.model.save(str(name))

def get_weights(network):
    weights = {}
    for idx, layer in enumerate(network.model.layers):
        #weights[idx] = layer.get_weights()[0]    
        if isinstance(layer,layers.Conv2D) or isinstance(layer,layers.Dense):
            weights[idx] = layer.get_weights()[0]
            continue
        weights[idx] = []
    return weights
