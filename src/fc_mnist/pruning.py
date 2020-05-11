import numpy as np

"""
    Different pruning methods used during the experiment, all methods will return a mask
"""

def prune(network, percents):
    weights = network.get_weights()
    mask = {}
    for idx, _ in enumerate(network.model.layers):
        if len(weights[idx]) <= 0:
            continue
        rows, cols = weights[idx].shape
        mask[idx] = np.ones((rows, cols))
        cutoff = np.round((rows*cols)*(1.0-percents[idx])).astype(int)
        flat_weights = weights[idx].flatten()
        flat_mask = mask[idx].flatten()
        partition = np.argpartition(np.abs(flat_weights), cutoff)[0:cutoff]
        print("Amount of zeros in mask: " + str(len(partition)) + " of " + str(rows*cols))
        flat_mask[partition] = 0
        mask[idx] = flat_mask.reshape((rows, cols))
    return mask