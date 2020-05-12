import tensorflow as tf
from constants import CIFAR10_DATA, SETTINGS_CONV2, SETTINGS_CONV4,SETTINGS_CONV6
from conv_models import CONV2_NETWORK, CONV4_NETWORK,CONV6_NETWORK
from tools import generate_percentages
from pruning import oneshot_pruning
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = CIFAR10_DATA.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
train_images = x_train / 255.0
test_images = x_test / 255.0

def iterative_test2():
    percents,iterations = generate_percentages([1.0,1.0,1.0],0.02,SETTINGS_CONV2['pruning_percentages'])
    
    og_network = CONV2_NETWORK(use_es = SETTINGS_CONV2['use_es'])

    #Save initial weights of the original matrix
    init_weights = og_network.get_weights()

    #Train original Network
    og_network.fit(x_train, y_train, SETTINGS_CONV2)

    #Evaluate original network
    test_loss, test_acc = og_network.evaluate_model(x_test,y_test)

    #Prune the network for x amount of iterations
    for i in range(0,iterations,2):
        print("Conv %: " + str(percents[i][0]) + ", Dense %: " + str(percents[i][1]) + ", Output %: " + str(percents[i][2]))
        mask = oneshot_pruning(og_network, percents[i][0],percents[i][1],percents[i][2])
        pruned_network = SETTINGS_CONV2(use_es = SETTINGS_CONV2['use_es'])
        pruned_network.fit_batch(x_train,y_train,mask,init_weights,SETTINGS_CONV2,x_test,y_test)
        pruned_network.evaluate_model(x_test,y_test)

        og_network = pruned_network

def iterative_test4():
    percents,iterations = generate_percentages([1.0,1.0,1.0],0.02,SETTINGS_CONV4['pruning_percentages'])

    og_network = CONV4_NETWORK(use_es = SETTINGS_CONV4['use_es'])
    
    #Save initial weights of the original matrix
    init_weights = og_network.get_weights()

    #Train original Network
    og_network.fit(x_train, y_train, SETTINGS_CONV4)

    #Evaluate original network
    test_loss, test_acc = og_network.evaluate_model(x_test,y_test)

    #Prune the network for x amount of times and evaluate each iteration
    for i in range(0,iterations,2):
        print("Conv %: " + str(percents[i][0]) + ", Dense %: " + str(percents[i][1]) + ", Output %: " + str(percents[i][2]))
        mask = oneshot_pruning(og_network, percents[i][0],percents[i][1],percents[i][2])
        pruned_network = CONV4_NETWORK(use_es =SETTINGS_CONV4['use_es'])
        pruned_network.fit_batch(x_train,y_train,mask,init_weights,SETTINGS_CONV4,x_test,y_test)
        pruned_network.evaluate_model(x_test,y_test)

        og_network = pruned_network
        
        
def iterative_test6():
    percents,iterations = generate_percentages([1.0,1.0,1.0],0.02,SETTINGS_CONV6['pruning_percentages'])
    for p in percents:
        print(percents[p])
    
    og_network = CONV6_NETWORK(use_es = SETTINGS_CONV6['use_es'])

    #Save initial weights of the original matrix
    init_weights = og_network.get_weights()

    #Train original network
    og_network.fit(x_train, y_train, SETTINGS_CONV6)

    #Evaluate original network
    test_loss, test_acc = og_network.evaluate_model(x_test,y_test)

    #Prune the network for x amount of times and evaluate each iteration
    for i in range(0,iterations,2):
        print("Conv %: " + str(percents[i][0]) + ", Dense %: " + str(percents[i][1]) + ", Output %: " + str(percents[i][2]))
        mask = oneshot_pruning(og_network, percents[i][0],percents[i][1],percents[i][2])
        pruned_network = CONV6_NETWORK(use_es = SETTINGS_CONV6['use_es'])
        pruned_network.fit_batch(x_train,y_train,mask,init_weights,SETTINGS_CONV6,x_test,y_test)
        pruned_network.evaluate_model(x_test,y_test)

        og_network = pruned_network


if __name__ == "__main__":
    iterative_test6()