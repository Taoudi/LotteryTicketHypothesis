import tensorflow as tf
from constants import MNIST_DATA
from fc_model import FC_NETWORK
import numpy as np
from tensorflow.keras import layers
from constants import PRUNING_PERCENTAGES

(x_train, y_train), (x_test, y_test) = MNIST_DATA.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.models.load_model("models/fc_mnist.h5")
network = FC_NETWORK(loaded_model=model)

print("EVALUATING ORIGINAL NETWORK: ")
network.evaluate_model(x_test, y_test)

#weights_init = np.load("weights_init.npz", allow_pickle=True)['weights_init']
weights_init = {}
for idx, layer in enumerate(network.model.layers):
    if len(layer.get_weights())>0:
        weights_init[idx] = layer.get_weights()[0]
    else:
        weights_init[idx] = []
#print(type(weights_init[1]))
#np.savez("weights_init.npz", weights_init=weights_init)
#network.fit(x_train, y_train, n_epochs=20)
#network.save_model("test1.h5")

masks = {}
for idx, layer in enumerate(network.model.layers):
    #print(weights_init[idx,:].shape)
    if len(weights_init[idx]) <= 0:
        continue
    masks[idx] = (np.random.random(weights_init[idx].shape) < PRUNING_PERCENTAGES[idx]).astype(int)
    #print(str(float(np.sum(r_matrix)/np.size(r_matrix))))

pruned_network = FC_NETWORK(use_earlyStopping=True)

print("Mask info: ")
for check in range(len(masks)):
    print(str(float(np.sum(masks[check+1])/np.size(masks[check+1]))))
print("--------------")

print(np.size(np.where(np.abs(pruned_network.model.get_weights()[0]) < 1e-6 )))

pruned_network.fit_batch(x_train, y_train, masks, weights_init, epochs=40)
print("EVALUATING PRUNED NETWORK (Pm = 20%)")
pruned_network.evaluate_model(x_test, y_test)

print(np.size(np.where(np.abs(pruned_network.model.get_weights()[0]) < 1e-6 )))
