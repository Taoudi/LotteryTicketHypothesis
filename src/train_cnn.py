import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

CIFAR10 = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = CIFAR10.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
X = train_images.reshape(np.size(train_images,0),3072)
testX = test_images.reshape(np.size(test_images,0),3072)

Y = np.zeros((train_labels.size, train_labels.max() + 1))
Y[np.arange(train_labels.size), train_labels] = 1

testY = np.zeros((test_labels.size, test_labels.max() + 1))
testY[np.arange(test_labels.size), test_labels] = 1

print(Y.shape)


def SVM(X, Y, test):
    clf = svm.SVC()
    clf.fit(X, Y)
    return clf.predict(test)

p = SVM(X[0:1000,:],Y[0:1000,:],testX[0:100,:])

summ = 0
for i, res in enumerate(p):
    if res == testY[i]:
        summ += 1

print(summ / np.size(p))
