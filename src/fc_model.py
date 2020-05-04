import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from constants import LAYERS, PRUNING_PERCENTAGES, OPTIMIZER_FC

class FC_NETWORK:
    def __init__(self, batch_size=60, use_earlyStopping=False):
        """
            Lenet-300-100 (Lenet-5) Architecture replica, 
            to use on both MNIST and Fashion MNIST
        """
        self.batch_size = batch_size
        self.pruning_rates = PRUNING_PERCENTAGES
        self.early_stopping = use_earlyStopping

        self.model = keras.Sequential()
        self.model.add(keras.layers.Flatten(input_shape=(28, 28)))

        for layer in LAYERS:
            (units, activation) = LAYERS[layer]
            self.model.add(keras.layers.Dense(units, activation=activation))
        
        self.model.compile(optimizer=OPTIMIZER_FC,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        
        if use_earlyStopping:
            self.es = EarlyStopping(monitor='val_loss', patience=10)
    
    def fit(self, data, labels, n_epochs=10):
        self.model.fit(x=data, y=labels, batch_size=self.batch_size, 
        validation_split=0.1 if self.early_stopping else None, epochs=n_epochs,
        callbacks=[self.es] if self.early_stopping else None)
        
        """
        Keras function fit
        fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, 
        validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, 
        sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, 
        validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)
        """

    def evaluate_model(self, test_data, test_labels):
        test_loss, test_acc = self.model.evaluate(test_data, test_labels, verbose=2)
        return test_loss, test_acc
    
    def save_model(self, filename):
        self.model.save('models/' + str(filename))