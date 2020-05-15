import tensorflow as tf
from tensorflow.keras import datasets,layers,models,initializers
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization

import numpy as np
import matplotlib.pyplot as plt

from constants import OPTIMIZER_CONV2, OPTIMIZER_CONV4, OPTIMIZER_CONV6
from tqdm import tqdm


class Network:
    def __init__(self,settings):
        self.batch_size = settings['batch_size']
        self.early_stopping = settings['use_es']
        self.settings= settings
        if settings['use_es']:
            self.es = EarlyStopping(monitor='val_loss', patience=settings['patience'])
    
    def fit(self,X,Y):
        history = self.model.fit(X, Y, epochs=self.settings['n_epochs'], callbacks=[self.es] if self.early_stopping else None, batch_size=self.batch_size,validation_split=self.settings['split'])
        return history

    def evaluate_model(self, testX, testY):
        test_loss, test_acc = self.model.evaluate(testX, testY, verbose=2)
        return test_loss, test_acc

    def save_(self,name):
        self.model.save(str(name))

    def get_weights(self):
        weights = {}
        for idx, layer in enumerate(self.model.layers):
            #weights[idx] = layer.get_weights()[0]
            if isinstance(layer,layers.Conv2D) or isinstance(layer,layers.Dense):
                weights[idx] = layer.get_weights()[0]
                continue
            weights[idx] = []
        return weights


    def shuffle_in_unison(self, x_train, y_train,split=0.1):
        """
            Shuffle the given data to both train and validation sets
        """
        n = np.size(x_train,axis=0)
        p = np.random.permutation(len(x_train))
        cutoff = int(split*n)
        p_val = p[0:cutoff]
        p_train = p[cutoff:]
        return x_train[p_train], y_train[p_train], x_train[p_val], y_train[p_val]

    def mask_weights(self, mask, weights):
        """
            Apply the mask on the weights, to disable their function
        """
        new_weights = list()
        for idx, layer in enumerate(self.model.layers):
            og_weight = layer.get_weights()
            if len(og_weight)>0:
                if isinstance(layer,layers.Dense) or isinstance(layer, layers.Conv2D):
                    new_weights.append(weights[idx]*mask[idx])
                    new_weights.append(layer.get_weights()[1])
                    
                    """elif isinstance(layer, layers.Conv2D):
                    row, col, kern, filt = weights[idx].shape
                    temp_weight = weights[idx]
                    for i,m in enumerate(mask[idx]):
                        if m == 0:
                            temp_weight[:,:,:,i] = np.zeros((row,col,kern))
                    new_weights.append(temp_weight)
                    new_weights.append(layer.get_weights()[1])"""
                else:
                    for w in layer.get_weights():
                        new_weights.append(w)
        return new_weights

    def fit_batch(self, data, labels, mask, weights_init, test_data=None,test_labels=None):
        """
            Train network with possibility of monitoring results each batch. The reasoning behind this is
            to be able to prune the weights accordingly.
        """
        if self.early_stopping:
            stop_patience = self.settings['patience']
        patience=0
        best_acc=0.0
        current_epoch = 0
        n = np.size(data,axis=0)
        n_batch = self.batch_size
        acc_history = []
        x_train, y_train, x_val, y_val = self.shuffle_in_unison(data,labels, self.settings['split'])

        if not self.settings['use_random_init']:
            current_weights = weights_init
        else:
            current_weights = self.get_weights()
        for e in range(0, self.settings['n_epochs']):
            x_train, y_train, _,_ = self.shuffle_in_unison(x_train,y_train,0.0)
            current_epoch=e
            print("Epoch " + str(e+1) + "/" + str(self.settings['n_epochs']))
            for j in tqdm(range(int(len(x_train) / n_batch))):
                masked_weights = self.mask_weights(mask, current_weights)
                self.model.set_weights(masked_weights)
                j_start = j*n_batch
                j_end = (j+1)*n_batch
                Xbatch = x_train[j_start:j_end,:,:]
                Ybatch = y_train[j_start:j_end]
                self.model.train_on_batch(Xbatch,Ybatch)
                current_weights = self.get_weights()
            if self.early_stopping:
                _, val_acc = self.evaluate_model(x_val,y_val)
                if val_acc <= best_acc:
                    patience=patience+1
                else:
                    best_acc=val_acc
            if self.settings['eval_test']:
                _, test_acc = self.evaluate_model(test_data,test_labels)
                acc_history.append(test_acc)
            if self.early_stopping:
                if patience>=stop_patience:
                    break
            
        new_weights = self.mask_weights(mask, current_weights)
        self.model.set_weights(new_weights)
        return acc_history, current_epoch


class CONV2_NETWORK(Network):
    def __init__(self,settings):
        dropout = settings['use_dropout']
        rate = settings['dropout_rate']
        self.model = models.Sequential()

        self.model.add(layers.BatchNormalization(input_shape=(32, 32, 3)))
        if dropout:
            self.model.add(layers.Dropout(rate))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.BatchNormalization())
        if dropout:
            self.model.add(layers.Dropout(rate))
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
        super().__init__(settings)



class CONV4_NETWORK(Network):
    def __init__(self, settings):
        self.model = models.Sequential()
        dropout = settings['use_dropout']
        rate = settings['dropout_rate']
        self.model.add(layers.BatchNormalization(input_shape=(32, 32, 3)))
        if dropout:
            self.model.add(layers.Dropout(rate))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.BatchNormalization())
        if dropout:
            self.model.add(layers.Dropout(rate))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.BatchNormalization())
        if dropout:
            self.model.add(layers.Dropout(rate))
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.BatchNormalization())
        if dropout:
            self.model.add(layers.Dropout(rate))
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(256,activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.Dense(256,activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.Dense(10))

        self.model.compile(OPTIMIZER_CONV4,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        super().__init__(settings)


class CONV6_NETWORK(Network):
    def __init__(self,settings):
        dropout = settings['use_dropout']
        rate = settings['dropout_rate']
        self.model = models.Sequential()
        self.model.add(layers.BatchNormalization(input_shape=(32, 32, 3)))
        if dropout:
            self.model.add(layers.Dropout(rate))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.BatchNormalization())
        if dropout:
            self.model.add(layers.Dropout(rate))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        
        self.model.add(layers.BatchNormalization())
        if dropout:
            self.model.add(layers.Dropout(rate))
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.BatchNormalization())
        if dropout:
            self.model.add(layers.Dropout(rate))
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        
        self.model.add(layers.BatchNormalization())
        if dropout:
            self.model.add(layers.Dropout(rate))
        self.model.add(layers.Conv2D(256, (3, 3), activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.BatchNormalization())
        if dropout:
            self.model.add(layers.Dropout(rate))
        self.model.add(layers.Conv2D(256, (3, 3), activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.MaxPooling2D((2, 2),padding='same'))

        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(256,activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.Dense(256,activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.Dense(10))


        self.model.compile(OPTIMIZER_CONV6,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
        super().__init__(settings)
