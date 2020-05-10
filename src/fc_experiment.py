import tensorflow as tf
from constants import MNIST_DATA
from fc_model import FC_NETWORK
import numpy as np
from tensorflow.keras import layers
from constants import PRUNING_PERCENTAGES, SETTINGS, LENET_PRUNE_FRACTIONS
from pruning import random_pruning, oneshot_pruning

(x_train, y_train), (x_test, y_test) = MNIST_DATA.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

#model = tf.keras.models.load_model("models/fc_mnist.h5")

def one_shot_pruning_experiment(pruning_rates):
    tot_acc = 0.0
    trials = SETTINGS['trials']
    results = np.zeros(trials)
    results_loss = np.zeros(trials)

    for i in range(0, trials):
        og_network = FC_NETWORK(i)
        og_network.fit(x_train,y_train,SETTINGS['n_epochs'])
        print("Evaluating original network")
        og_network.evaluate_model(x_test, y_test)
        #print(og_network.get_summary())
        #print("Creating the pruned network")
        #mask = random_pruning(og_network)
        mask = oneshot_pruning(og_network, pruning_rates)
        pruned_network = FC_NETWORK(i)
        pruned_network.fit_batch(x_train, y_train, mask, og_network.weights_init, SETTINGS, x_test, y_test)
        print("Evaluating the pruned network")
        test_loss, test_acc = pruned_network.evaluate_model(x_test, y_test)
        tot_acc+=test_acc
        print("Iteration: " + str(i) + ", Test Accuracy: " + str(test_acc))
        results[i] = test_acc
        results_loss[i] = test_loss

    tot_acc=float(tot_acc/trials)
    print("Total Average Accuracy: " + str(tot_acc))
    print(results)
    print(results_loss)
    return tot_acc
    #np.savez("data/OneShotPruningManyTrialsAcc.npz", histories=results)
    #np.savez("data/OneShotPruningManyTrialsLoss.npz", histories=results_loss)



def iterative_pruning_experiment():
    trials = SETTINGS['trials']
    iterations = SETTINGS['prune_iterations']
    
    histories = np.zeros((iterations+1, SETTINGS['n_epochs']))

    for k in range(0, trials):
        print("TRIAL " + str(k+1) + "/" + str(trials))
        og_network = FC_NETWORK()

        S = [0.0, 1.0, 1.0, 0.5] # Base case
        c = 1-PRUNING_PERCENTAGES[1]**(1/iterations)
        mask = oneshot_pruning(og_network,S)
        acc_history = og_network.fit_batch(x_train, y_train, mask, og_network.weights_init, SETTINGS, x_test, y_test)
        histories[iterations,:] += np.asarray(acc_history)

        for i in range(0,iterations):
            for j,s in enumerate(S):
                S[j] = S[j] - S[j]*c

            print("Prune iteration: " + str(i+1) + ", S: " + str(S))
            print("Creating the pruned network")
            mask = oneshot_pruning(og_network, S)
            pruned_network = FC_NETWORK()
            acc_history = pruned_network.fit_batch(x_train, y_train, mask, og_network.weights_init, SETTINGS, x_test, y_test)
            histories[i,:] += np.asarray(acc_history)

            og_network = pruned_network
            
    histories = histories/trials
    np.savez("data/iterpr_lenet_100it_5trials.npz", histories=histories)


def big_one_shot_pruning_experiment():
    trials = SETTINGS['trials']
    iterations = SETTINGS['prune_iterations']

    histories = np.zeros((iterations+1,2))
    S = [0.0, 1.0, 1.0, 1.0] # Base case
    c = 1-PRUNING_PERCENTAGES[1]**(1/iterations)
    #
    # c=0.2
    c2 = c/2

    for i in range(0,iterations):
        for j,s in enumerate(S):
            if j ==len(S)-1:
                S[j] = S[j] - S[j]*c2
            else:
                S[j] = S[j] - S[j]*c
        print(S)
    tot_acc = one_shot_pruning_experiment(S)
    histories[iterations,0] = S[1]
    histories[iterations,1] = tot_acc
    c = 1-PRUNING_PERCENTAGES[1]**(1/iterations)
    #c=0.2
    c2 = c/2

    for i in range(0,iterations):
        for j,s in enumerate(S):
            if j ==len(S)-1:
                S[j] = S[j] - S[j]*c2
            else:
                S[j] = S[j] - S[j]*c
        print(S)
    
        tot_acc = one_shot_pruning_experiment(S)
        histories[i,0] = S[1]
        histories[i,1] = tot_acc

    print("HERE")
    print(histories)
    np.savez("data/OneShotPruningDifferentRates.npz", histories=histories)
#iterative_pruning_experiment()
big_one_shot_pruning_experiment()
