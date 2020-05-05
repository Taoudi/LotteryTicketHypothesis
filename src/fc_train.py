from constants import MNIST_DATA
from fc_model import FC_NETWORK

"""
    Training different types of networks, with and without pruning
"""

def train_vanilla(x_train, y_train, save_model=False):
    """
        Trains a keras sequential model without any pruning
    """
    network = FC_NETWORK(use_earlyStopping=True)
    network.fit(x_train, y_train, n_epochs=20)
    if save_model:
        print(">> Model Parameters Saved as fc_mnist.h5")
        network.save_model("fc_mnist.h5")