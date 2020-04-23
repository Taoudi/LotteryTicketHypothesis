import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


MNIST = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = MNIST.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0
class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

class Simple_MLP:
    def __init__(self):
        pass

    


