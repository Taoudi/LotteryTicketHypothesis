from constants import PRUNING_PERCENTAGES
import numpy as np

"""
    Different pruning methods used during the experiment, all methods will return a mask
"""

def iterative_pruning():
    pass

def oneshot_pruning(network):
    weights = network.get_weights()
    mask = {}
    for idx, layer in enumerate(network.model.layers):
        if len(weights[idx]) <= 0:
            continue
        percent = 1.0 - PRUNING_PERCENTAGES[idx]
        rows, cols = weights[idx].shape
        mask[idx] = np.ones((rows, cols))
        k = np.round((rows*cols)*percent).astype(int)
        flat_weights = weights[idx].flatten()
        flat_mask = mask[idx].flatten()
        partition = np.argpartition(np.abs(flat_weights),k)[0:k]
        flat_mask[partition] = 0
        mask[idx] = flat_mask.reshape((rows,cols))
    return mask

def random_pruning(network):
    weights = network.get_weights()
    mask = {}
    for idx, layer in enumerate(network.model.layers):
        if len(weights[idx]) <= 0:
            continue
        mask[idx] = (np.random.random(weights[idx].shape) < PRUNING_PERCENTAGES[idx]).astype(int)
    return mask