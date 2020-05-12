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

def simple_test2():
    network = CONV2_NETWORK()
    network.fit(x_train, y_train, SETTINGS_CONV2)
    test_loss, test_acc = network.evaluate_model(x_test,y_test)
    network.model.summary()
    print(test_acc)

def simple_test4():
    network = CONV4_NETWORK()
    #init_weights = get_weights(network)
    #print(network.model.get_weights())
    network.fit(x_train, y_train, SETTINGS_CONV4)
    test_loss, test_acc = network.evaluate_model(x_test,y_test)
    #network.model.summary()
    print(test_acc)

def iterative_test2():
    percents,iterations = generate_percentages([1.0,1.0,1.0],0.02)
    iterations = 4
    og_network = CONV2_NETWORK()
    init_weights = og_network.get_weights()
    og_network.fit(x_train, y_train, SETTINGS_CONV2)
    test_loss, test_acc = og_network.evaluate_model(x_test,y_test)
    for i in range(0,iterations,2):
        print("Dense %: " + str(percents[i][1]))
        print("Conv/Output %: " + str(percents[i][0]))
        mask = oneshot_pruning(og_network, percents[i][0],percents[i][1],percents[i][0])
        pruned_network = SETTINGS_CONV2()
        pruned_network.fit_batch(x_train,y_train,mask,init_weights,SETTINGS_CONV2,x_test,y_test)
        pruned_network.evaluate_model(x_test,y_test)

        og_network = pruned_network

def iterative_test4():
    percents,iterations = generate_percentages([1.0,1.0,1.0],0.02)
    iterations = 4
    og_network = CONV4_NETWORK()
    init_weights = og_network.get_weights()
    og_network.fit(x_train, y_train, SETTINGS_CONV4)
    test_loss, test_acc = og_network.evaluate_model(x_test,y_test)
    for i in range(0,iterations,2):
        print("Dense %: " + str(percents[i][1]))
        print("Conv/Output %: " + str(percents[i][0]))
        mask = oneshot_pruning(og_network, percents[i][0],percents[i][1],percents[i][0])
        pruned_network = CONV4_NETWORK()
        pruned_network.fit_batch(x_train,y_train,mask,init_weights,SETTINGS_CONV4,x_test,y_test)
        pruned_network.evaluate_model(x_test,y_test)

        og_network = pruned_network
        
        
def iterative_test6():
    percents,iterations = generate_percentages([1.0,1.0,1.0],0.02)
    iterations = 4
    og_network = CONV6_NETWORK(SETTINGS_CONV6['early_stopping'])
    init_weights = og_network.get_weights()
    og_network.fit(x_train, y_train, SETTINGS_CONV6)
    test_loss, test_acc = og_network.evaluate_model(x_test,y_test)
    for i in range(0,iterations,2):
        print("Dense %: " + str(percents[i][1]))
        print("Conv/Output %: " + str(percents[i][0]))
        mask = oneshot_pruning(og_network, percents[i][0],percents[i][1],percents[i][0])
        pruned_network = CONV6_NETWORK()
        pruned_network.fit_batch(x_train,y_train,mask,init_weights,SETTINGS_CONV6,x_test,y_test)
        pruned_network.evaluate_model(x_test,y_test)

        og_network = pruned_network

    #_,epoch = og_network.fit_batch(x_train, y_train, mask, init_weights, SETTINGS_CONV2, x_test, y_test)
    #test_loss,test_acc = og_network.evaluate_model(x_test, y_test)
    #network.model.summary()
if __name__ == "__main__":
    iterative_test6()