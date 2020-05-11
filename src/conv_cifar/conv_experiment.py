import tensorflow as tf
from constants import CIFAR10_DATA, SETTINGS_CONV2, SETTINGS_CONV4
from conv_models import CONV2_NETWORK, CONV4_NETWORK, get_weights
from tools import generate_percentages
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


def simple_oneshot_test4():
    percents,iterations = generate_percentages([1.0,1.0,1.0],0.02)
    print(percents[14])
    og_network = CONV4_NETWORK()
    init_weights = get_weights(og_network)
    #og_network.fit(x_train, y_train, SETTINGS_CONV4)
    #test_loss, test_acc = og_network.evaluate_model(x_test,y_test)
    
    #mask = prune(og_network, percents)
    #_,epoch = og_networks[i].fit_batch(x_train, y_train, mask, og_network.weights_init, SETTINGS, x_test, y_test)
    #test_loss,test_acc = og_network.evaluate_model(x_test, y_test)


    #network.model.summary()
    #print(test_acc)
if __name__ == "__main__":
    simple_oneshot_test4()