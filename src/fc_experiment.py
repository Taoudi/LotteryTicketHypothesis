import tensorflow as tf
from constants import MNIST_DATA
from fc_model import FC_NETWORK
import numpy as np
from tensorflow.keras import layers
from constants import PRUNING_PERCENTAGES
from pruning import random_pruning, oneshot_pruning

(x_train, y_train), (x_test, y_test) = MNIST_DATA.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

#model = tf.keras.models.load_model("models/fc_mnist.h5")

tot_acc = 0.0
trials = 20
for i in range(0, trials):
    og_network = FC_NETWORK(use_earlyStopping=True)
    og_network.fit(x_train,y_train,20)
    print("Evaluating original network")
    og_network.evaluate_model(x_test, y_test)
    #print(og_network.get_summary())
    #print("Creating the pruned network")
    #mask = random_pruning(og_network)
    mask = oneshot_pruning(og_network)
    pruned_network = FC_NETWORK(use_earlyStopping=True)
    pruned_network.fit_batch(x_train, y_train, mask, og_network.weights_init, epochs=15)
    print("Evaluating the pruned network")
    test_loss, test_acc = pruned_network.evaluate_model(x_test, y_test)
    tot_acc+=test_acc

tot_acc=float(tot_acc/trials)

