import tensorflow as tf
from constants import CIFAR10_DATA, SETTINGS_CONV2, SETTINGS_CONV4, SETTINGS_CONV6, TRIALS
from conv_models import CONV2_NETWORK, CONV4_NETWORK,CONV6_NETWORK
from tools import generate_percentages
from pruning import prune
import numpy as np
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = CIFAR10_DATA.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255.0
x_test = x_test / 255.0

def iterative_test_conv(settings, network_type=2,filename=""):
    percents, iterations = generate_percentages([1.0,1.0,1.0],0.02,settings['pruning_percentages'])
    histories = np.zeros(iterations+1)
    es_epochs = np.zeros(iterations+1)
  
    for trial in range(TRIALS):
        if network_type == 2:
            og_network = CONV2_NETWORK(settings)
        elif network_type == 4:
            og_network = CONV4_NETWORK(settings)
        elif network_type == 6:
            og_network = CONV6_NETWORK(settings)
        
        #Save initial weights of the original matrix
        init_weights = og_network.get_weights()

        #Train original Network
        mask = prune(og_network, 1.0, 1.0, 1.0)

        _, epoch = og_network.fit_batch(x_train, y_train, mask, init_weights, x_test, y_test)
        es_epochs[0] += epoch

        #Evaluate original network and save results
        _, test_acc = og_network.evaluate_model(x_test,y_test)
        histories[0] += test_acc

        #Prune the network for x amount of iterations, evaulate each iteration and save results
        for i in range(0,iterations):
            print("Conv %: " + str(percents[i][0]) + ", Dense %: " + str(percents[i][1]) + ", Output %: " + str(percents[i][2]))
            mask = prune(og_network, percents[i][0],percents[i][1],percents[i][2])
            #for w in masked_weights:
            #    print(np.count_nonzero(w==0)/np.size(w))


            if network_type == 2:
                pruned_network = CONV2_NETWORK(settings)
            elif network_type == 4:
                pruned_network = CONV4_NETWORK(settings)
            elif network_type == 6:
                pruned_network = CONV6_NETWORK(settings)
            
            _, epoch = pruned_network.fit_batch(x_train,y_train,mask,init_weights,x_test,y_test)
            _, test_acc = pruned_network.evaluate_model(x_test,y_test)
            histories[i+1] += test_acc
            es_epochs[i+1] += epoch

            og_network = pruned_network
        
        filename += "_trial-" + str(trial+1)
        
        print(histories)
        print(es_epochs)
        np.savez(filename + ".npz", histories=histories, es_epochs=es_epochs)
    return histories, es_epochs


if __name__ == "__main__":
    network_type = 6
    settings = SETTINGS_CONV6

    uses_reinit = settings['use_random_init']
    uses_dropout = settings['use_dropout']
    uses_rate = settings['dropout_rate']
    uses_es = settings['use_es']

    filename = "conv" + str(network_type) + "_rand-" + str(uses_reinit) + "_es-" + str(uses_es) + "_dp-" + str(uses_dropout) + "_rate-" + str(uses_rate)
    histories, es_epochs = iterative_test_conv(settings, network_type,filename)