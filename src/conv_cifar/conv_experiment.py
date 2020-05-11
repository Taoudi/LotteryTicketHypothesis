import tensorflow as tf
from constants import CIFAR10_DATA, SETTINGS_CONV2, SETTINGS_CONV4
from conv_models import CONV2_NETWORK, CONV4_NETWORK

(x_train, y_train), (x_test, y_test) = CIFAR10_DATA.load_data()

train_images = x_train / 255.0
test_images = x_test / 255.0

def simple_test():
    network = CONV2_NETWORK()
    network.fit(x_train, y_train, SETTINGS_CONV2)
    test_loss, test_acc = network.evaluate_model(x_test,y_test)
    print(test_acc)


if __name__ == "__main__":
    simple_test()