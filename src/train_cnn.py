import tensorflow as tf
from tensorflow.keras import datasets,layers,models,initializers
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


class Simple_CNN:
    def __init__(self):
        self.model = models.Sequential()

        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3),kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Conv2D(256, (3, 3), activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Conv2D(256, (3, 3), activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.MaxPooling2D((2, 2),padding='same'))



        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.Dense(256, activation='relu',kernel_initializer=initializers.glorot_normal(seed=None)))
        self.model.add(layers.Dense(10))
        #self.model.add(layers.Softmax())

        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        self.es = EarlyStopping(monitor='val_loss', patience=8)
        self.epochs = 40
        self.bs = 60

    def fit_(self,X,Y):
        history = self.model.fit(X, Y, epochs=self.epochs, validation_split=0.1,callbacks=[self.es],batch_size=self.bs, workers=1, use_multiprocessing=False) ,
        return history

    def eval_(self, testX, testY):
        test_loss, test_acc = self.model.evaluate(testX, testY, verbose=2)
        return test_loss, test_acc

    def save_(self,name):
        self.model.save(str(name))

CIFAR10 = datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = CIFAR10.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
cnn = Simple_CNN()
cnn.fit_(train_images, train_labels)
loss, acc = cnn.eval_(test_images,test_labels)
print(acc)
cnn.save_("CIFAR.h5")

    #X = train_images.reshape(np.size(train_images,0),3072)
    #testX = test_images.reshape(np.size(test_images,0),3072)

    #Y = np.zeros((train_labels.size, train_labels.max() + 1))
    #Y[np.arange(train_labels.size), train_labels] = 1

    #testY = np.zeros((test_labels.size, test_labels.max() + 1))
    #testY[np.arange(test_labels.size), test_labels] = 1











def SVM(X, Y, testX,testY):
    clf = svm.SVC()
    clf.fit(X, Y)
    summ = 0
    p = clf.predict(test)
    for i, res in enumerate(p):
        #print(res, test_labels[i])
        if res == test_labels[i]:
            summ += 1

    return(summ / np.size(p))



