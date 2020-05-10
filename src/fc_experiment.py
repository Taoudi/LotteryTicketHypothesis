import tensorflow as tf
from constants import MNIST_DATA
from fc_model import FC_NETWORK
import numpy as np
from tensorflow.keras import layers
from constants import PRUNING_PERCENTAGES, SETTINGS
from pruning import random_pruning, oneshot_pruning, prune

(x_train, y_train), (x_test, y_test) = MNIST_DATA.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

#model = tf.keras.models.load_model("models/fc_mnist.h5")

def one_shot_pruning_experiment():
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
        mask = oneshot_pruning(og_network, PRUNING_PERCENTAGES)
        pruned_network = FC_NETWORK(use_earlyStopping=True)
        pruned_network.fit_batch(x_train, y_train, mask, og_network.weights_init, SETTINGS)
        print("Evaluating the pruned network")
        _, test_acc = pruned_network.evaluate_model(x_test, y_test)
        tot_acc+=test_acc

    tot_acc=float(tot_acc/trials)
    print("Total Average Accuracy: " + str(tot_acc))

def generate_percentages(base_percents, lower_bound):
    percentages = {}
    percents = base_percents
    idx = 0
    while percents[1] >= lower_bound:
        new_percents = []
        for i in range(len(base_percents)):
            new_percents.append((1-PRUNING_PERCENTAGES[i])*percents[i])
        
        percentages[idx] = new_percents
        percents = new_percents
        
        idx += 1
    
    return percentages, len(percentages)

def iterative_pruning_experiment():
    trials = SETTINGS['trials']
    #iterations = SETTINGS['prune_iterations']
    percents = [0.0, 1.0, 1.0, 1.0]
    percentages, iterations = generate_percentages(percents, SETTINGS['lower_bound'])
    histories = np.zeros((iterations+1, SETTINGS['n_epochs']))

    for k in range(0, trials):
        print("TRIAL " + str(k+1) + "/" + str(trials))
        og_network = FC_NETWORK()

        mask = prune(og_network, percents)
        acc_history = og_network.fit_batch(x_train, y_train, mask, og_network.weights_init, SETTINGS, x_test, y_test)
        histories[iterations,:] += np.asarray(acc_history)
        
        for i in range(0,iterations):
            S = percentages[i]

            print("Prune iteration: " + str(i+1) + "/" + str(iterations) + ", S: " + str(S))
            print("Creating the pruned network")
            mask = prune(og_network, S)
            pruned_network = FC_NETWORK()
            acc_history = pruned_network.fit_batch(x_train, y_train, mask, og_network.weights_init, SETTINGS, x_test, y_test)
            histories[i,:] += np.asarray(acc_history)

            og_network = pruned_network
        
            
            
    histories = histories/trials
    np.savez("data/iterpr_lenet_20perc.npz", histories=histories)

iterative_pruning_experiment()

