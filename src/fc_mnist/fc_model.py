import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from constants import LAYERS, PRUNING_PERCENTAGES, OPTIMIZER_FC
from tqdm import tqdm
import numpy as np

class FC_NETWORK:
    def get_weights(self):
        """
            Retrieve the weights of the network
        """
        weights = {}
        for idx, layer in enumerate(self.model.layers):
            if len(layer.get_weights())>0:
                weights[idx] = layer.get_weights()[0]
            else:
                weights[idx] = []                                
        return weights

    def __init__(self,seed=None,batch_size=60, use_earlyStopping=False, loaded_model=None):
        """
            Fully-connected Lenet-300-100 (Lenet-5) Network Architecture
        """
        self.batch_size = batch_size
        self.pruning_rates = PRUNING_PERCENTAGES
        self.early_stopping = use_earlyStopping

        if loaded_model == None:
            self.model = keras.Sequential()
            self.model.add(keras.layers.Flatten(input_shape=(28, 28)))
            for layer in LAYERS:
                (units, activation) = LAYERS[layer]
                self.model.add(keras.layers.Dense(units, activation=activation, kernel_initializer=tf.keras.initializers.glorot_normal(seed=None)))
        
        
            self.model.compile(optimizer=OPTIMIZER_FC,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

            self.weights_init = self.get_weights()
        
        else:
            self.model = loaded_model

        if use_earlyStopping:
            self.es = EarlyStopping(monitor='val_loss', patience=10)
        
    
    def fit(self, data, labels, n_epochs=20):
        """
            Train network with given data and labels, number of epochs is 20 by default
        """
        self.model.fit(x=data, y=labels, batch_size=self.batch_size, 
        validation_split=0.1 if self.early_stopping else None, epochs=n_epochs,
        callbacks=[self.es] if self.early_stopping else None)

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
            if len(layer.get_weights())>0:
                new_weights.append(weights[idx]*mask[idx])
                new_weights.append(layer.get_weights()[1])
            else:
                continue
        return new_weights
            
    def fit_batch(self, data, labels, mask, weights_init, settings, test_data=None,test_labels=None):
        """
            Train network with possibility of monitoring results each batch. The reasoning behind this is
            to be able to prune the weights accordingly.
        """
        if self.early_stopping:
            stop_patience = settings['patience']
        patience=0
        best_acc=0.0
        current_epoch = 0
        n = np.size(data,axis=0)
        n_batch = self.batch_size
        acc_history = []
        x_train, y_train, x_val, y_val = self.shuffle_in_unison(data,labels, settings['split'])

        if not settings['use_random_init']:
            current_weights = weights_init
        else:
            current_weights = self.get_weights()
        for e in range(0, settings['n_epochs']):
            x_train, y_train, _,_ = self.shuffle_in_unison(x_train,y_train,0.0)
            current_epoch=e
            print("Epoch " + str(e+1) + "/" + str(settings['n_epochs']))
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
            if settings['eval_test']:
                _, test_acc = self.evaluate_model(test_data,test_labels)
                acc_history.append(test_acc)
            if self.early_stopping:
                if patience>=stop_patience:
                    break
            
        new_weights = self.mask_weights(mask, current_weights)
        self.model.set_weights(new_weights)
        return acc_history, current_epoch

    def evaluate_model(self, test_data, test_labels,verbose=2):
        """
            Evaluate the model given the test data (unseen data)
        """
        test_loss, test_acc = self.model.evaluate(test_data, test_labels, verbose=verbose)
        return test_loss, test_acc
    
    def save_model(self, filename):
        """
            Helper-function to save the model
        """
        self.model.save('models/' + str(filename))

    def get_summary(self):
        """
            Helper-function to get the summary of the network architecture
        """
        return self.model.summary()

            
