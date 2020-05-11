import tensorflow as tf
from constants import MNIST_DATA
from fc_model import FC_NETWORK
import numpy as np
from tensorflow.keras import layers
from constants import PRUNING_PERCENTAGES, SETTINGS
from pruning import random_pruning, oneshot_pruning, prune
from tools import generate_percentages

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

def iterative_pruning_experiment():
    trials = SETTINGS['trials']
    #iterations = SETTINGS['prune_iterations']
    percents = [0.0, 1.0, 1.0, 1.0]
    percentages, iterations = generate_percentages(percents, SETTINGS['lower_bound'])
    histories = np.zeros((trials, iterations+1, SETTINGS['n_epochs']))
    es_epochs = np.zeros((trials, iterations+1))
    for k in range(0, trials):
        print("TRIAL " + str(k+1) + "/" + str(trials))
        og_network = FC_NETWORK(use_earlyStopping=True)

        mask = prune(og_network, percents)
        acc_history, epoch = og_network.fit_batch(x_train, y_train, mask, og_network.weights_init, SETTINGS, x_test, y_test)
        histories[k, iterations, 0:epoch+1] = np.asarray(acc_history)
        es_epochs[k, iterations] = epoch
        for i in range(0,iterations):
            S = percentages[i]

            print("Prune iteration: " + str(i+1) + "/" + str(iterations) + ", S: " + str(S))
            print("Creating the pruned network")
            mask = prune(og_network, S)
            pruned_network = FC_NETWORK(use_earlyStopping=True)
            acc_history, epoch = pruned_network.fit_batch(x_train, y_train, mask, og_network.weights_init, SETTINGS, x_test, y_test)
            histories[k, i, 0:epoch+1] = np.asarray(acc_history)
            es_epochs[k, i] = epoch
            og_network = pruned_network
    
    #histories = histories/trials
    #es_epochs = float(es_epochs/trials)
    #np.savez("data/iterpr_lenet_20perc_reinit_es.npz", histories=histories, es_epochs=es_epochs)


    
def big_one_shot_pruning_experiment():
    """
    - For multiple trials:
        - train original network and evaluate results
        - find mask depending on weights of og network 
        - apply mask to a new network and disable corresponding weights
        - train the new pruned network and evaluate results
    """
    percentages, iterations = generate_percentages([0.0, 1.0, 1.0, 1.0], SETTINGS['lower_bound'])
    #print(percentages)
    #print(iterations)
    trials = SETTINGS['trials']
    og_networks = list()
    tot_acc = np.zeros(iterations+1)
    tot_loss = np.zeros(iterations+1)
    tot_epoch = np.zeros(iterations+1)

    # Training and evaluating the unpruned network over multiple trials
    print("Training and Evaluating OG networks")
    percents = [0.0, 1.0, 1.0, 1.0]

    for i in range(0,trials):
        print("TRIAL " + str(i+1) + "/" + str(trials))
        og_networks.append(FC_NETWORK(use_earlyStopping=True))
        mask = prune(og_networks[i], percents)
        _,epoch = og_networks[i].fit_batch(x_train, y_train, mask, og_networks[i].weights_init, SETTINGS, x_test, y_test)
        test_loss,test_acc = og_networks[i].evaluate_model(x_test, y_test)
        tot_acc[0]+=test_acc
        tot_loss[0]+=test_loss
        tot_epoch[0]+=epoch
        print(epoch)
    tot_acc[0]=float(tot_acc[0]/trials)
    tot_loss[0]=float(tot_loss[0]/trials)
    tot_epoch[0] = float(tot_epoch[0]/trials)
    # Training and evaluating pruned networks of different pruning rates over multiple trials
    for j in range(1,iterations+1):
        print("Training and Evaluating pruned networks, iteration: " + str(j) + "/" + str(iterations))
        print("Percentage: " + str(percentages[j-1]))
        for og in og_networks:
            mask = oneshot_pruning(og, percentages[j-1])
            pruned_network = FC_NETWORK(use_earlyStopping=True)
            _,epoch = pruned_network.fit_batch(x_train, y_train, mask, og.weights_init, SETTINGS, x_test, y_test)
            print(epoch)
            test_loss, test_acc = pruned_network.evaluate_model(x_test, y_test)
            tot_acc[j]+=test_acc
            tot_loss[j]+=test_loss
            tot_epoch[j]+=epoch

        tot_acc[j]=float(tot_acc[j]/trials)
        tot_loss[j]=float(tot_loss[j]/trials)
        tot_epoch[j] = float(tot_epoch[j]/trials)
        print(tot_epoch)

    print(tot_acc)
    print(tot_loss)
    print(tot_epoch)
    np.savez("OneShotPruningAcc_5trials_50epochs_20perc_ES_rand.npz", histories=tot_acc)
    np.savez("OneShotPruningLoss_5trials_50epochs_20perc_ES_rand.npz", histories=tot_loss)
    np.savez("OneShotPruningEpochs_5trials_50epochs_ES_rand.npz", histories=tot_epoch)
