import tensorflow as tf
from constants import MNIST_DATA
from fc_model import FC_NETWORK
import numpy as np
from tensorflow.keras import layers
from constants import PRUNING_PERCENTAGES, SETTINGS
from pruning import prune
from tools import generate_percentages

(x_train, y_train), (x_test, y_test) = MNIST_DATA.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

def iterative_pruning_experiment():
    """
    - For multiple trials:
        - Create the original network
        - Train original network, save results
        - For multiple iterations:
            - Create mask with the pruning precentages generated by generate_percentages function in tools.py
            - Prune the network with the pruning percentages
            - Train the pruned network, save results
            - Set the original network to be the pruned network
    """
    trials = SETTINGS['trials']
    percents = [0.0, 1.0, 1.0, 1.0]
    percentages, iterations = generate_percentages(percents, SETTINGS['lower_bound'])
    histories = np.zeros((iterations+1, SETTINGS['n_epochs']))
    #es_epochs = np.zeros((trials, iterations+1))
    for k in range(0, trials):
        print("TRIAL " + str(k+1) + "/" + str(trials))
        og_network = FC_NETWORK(use_earlyStopping=SETTINGS['use_es'])
        init_weights = og_network.weights_init
        mask = prune(og_network, percents)
        acc_history, _ = og_network.fit_batch(x_train, y_train, mask, init_weights, SETTINGS, x_test, y_test)
        histories[iterations] += np.asarray(acc_history)
        #es_epochs[k, iterations] = epoch
        for i in range(0,iterations):
            S = percentages[i]

            print("Prune iteration: " + str(i+1) + "/" + str(iterations) + ", S: " + str(S))
            print("Creating the pruned network")
            mask = prune(og_network, S)
            pruned_network = FC_NETWORK(use_earlyStopping=SETTINGS['use_es'])
            acc_history, _ = pruned_network.fit_batch(x_train, y_train, mask, init_weights, SETTINGS, x_test, y_test)
            histories[i] += np.asarray(acc_history)
            #es_epochs[k, i] = epoch
            og_network = pruned_network
    
    histories = histories/trials
    #es_epochs = float(es_epochs/trials)
    np.savez("data/iterpr_lenet_20perc.npz", histories=histories)


    
def one_shot_pruning_experiment():
    """
    - For multiple trials:
        - Train original network and evaluate results
        - Find mask depending on weights of original network 
        - Apply mask to a new network and disable corresponding weights
        - Train the new pruned network and evaluate results
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
        og_networks.append(FC_NETWORK(use_earlyStopping=SETTINGS['use_es']))
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
            mask = prune(og, percentages[j-1])
            pruned_network = FC_NETWORK(use_earlyStopping=SETTINGS['use_es'])
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
    #np.savez("data/os_acc_.npz", histories=tot_acc)
    #np.savez("data/OneShotPruningLoss_5trials_20epochs_20perc_random.npz", histories=tot_loss)
    #np.savez("data/OneShotPruningEpochs_5trials_20epochs_20perc_random.npz", histories=tot_epoch)
one_shot_pruning_experiment()
#iterative_pruning_experiment()