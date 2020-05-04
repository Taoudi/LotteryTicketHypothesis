from constants import MNIST_DATA
from fc_model import FC_NETWORK
from keras.models import load_model

(x_train, y_train), (x_test, y_test) = MNIST_DATA.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

network = FC_NETWORK(use_earlyStopping=True)
network.fit(x_train, y_train, n_epochs=20)
test_loss, test_acc = network.evaluate_model(x_test, y_test)
print("\nTest Accuracy: " + str(test_acc))
if test_acc > 0.9815:
    network.save_model("fc_mnist.h5")