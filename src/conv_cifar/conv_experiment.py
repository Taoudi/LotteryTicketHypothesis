import tensorflow as tf
from constants import CIFAR10_DATA, SETTINGS_CONV2, SETTINGS_CONV4, SETTINGS_CONV6
from conv_models import CONV2_NETWORK, CONV4_NETWORK,CONV6_NETWORK
from tools import generate_percentages
from pruning import prune
import numpy as np

(x_train, y_train), (x_test, y_test) = CIFAR10_DATA.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

train_images = x_train / 255.0
test_images = x_test / 255.0

def iterative_test_conv(settings, network_type=2):
    percents, iterations = generate_percentages([1.0,1.0,1.0],0.02,settings['pruning_percentages'])
    histories = np.zeros(iterations+1)
    es_epochs = np.zeros(iterations+1)

    if network_type == 2:
        og_network = CONV2_NETWORK(dropout=settings['use_dropout'], use_es = settings['use_es'], patience=settings['patience'])
    elif network_type == 4:
        og_network = CONV4_NETWORK(dropout=settings['use_dropout'], use_es = settings['use_es'], patience=settings['patience'])
    elif network_type == 6:
        og_network = CONV6_NETWORK(dropout=settings['use_dropout'], use_es = settings['use_es'], patience=settings['patience'])
    
    #Save initial weights of the original matrix
    init_weights = og_network.get_weights()

    #Train original Network
    mask = prune(og_network, 1.0, 1.0, 1.0)
    _, epoch = og_network.fit_batch(x_train, y_train, mask, init_weights, settings, x_test, y_test)
    es_epochs[0] = epoch

    #Evaluate original network and save results
    _, test_acc = og_network.evaluate_model(x_test,y_test)
    histories[0] = test_acc

    #Prune the network for x amount of iterations, evaulate each iteration and save results
    for i in range(0,iterations):
        print("Conv %: " + str(percents[i][0]) + ", Dense %: " + str(percents[i][1]) + ", Output %: " + str(percents[i][2]))
        mask = prune(og_network, percents[i][0],percents[i][1],percents[i][2])
        if network_type == 2:
            pruned_network = CONV2_NETWORK(dropout=settings['use_dropout'], use_es = settings['use_es'])
        elif network_type == 4:
            pruned_network = CONV4_NETWORK(dropout=settings['use_dropout'], use_es = settings['use_es'])
        elif network_type == 6:
            pruned_network = CONV6_NETWORK(dropout=settings['use_dropout'], use_es = settings['use_es'])
        
        _, epoch = pruned_network.fit_batch(x_train,y_train,mask,init_weights,settings,x_test,y_test)
        _, test_acc = pruned_network.evaluate_model(x_test,y_test)
        histories[i+1] = test_acc
        es_epochs[i+1] = epoch

        og_network = pruned_network
    
    return histories, es_epochs


if __name__ == "__main__":
    network_type = 6
    settings = SETTINGS_CONV6

    histories, es_epochs = iterative_test_conv(settings, network_type)

    print(histories)
    print(es_epochs)

    uses_es = settings['use_es']
    uses_reinit = settings['use_random_init']
    filename = "data/conv" + str(network_type) + "_rand-" + str(uses_reinit) + "_es-" + str(uses_es) + "_data.npz"
    np.savez(filename, histories=histories, es_epochs=es_epochs)