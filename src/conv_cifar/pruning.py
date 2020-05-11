
import numpy as np

def oneshot_pruning(network,conv_percent,dense_percent,output_percent):
    weights = network.get_weights()
    mask = {}
    for idx, layer in enumerate(network.model.layers):
        percent = 0
        if isinstance(layer,layers.Conv2D):
            percent = conv_percent
        elif(idx == len(network.model.layers)-1):
            percent = output_percent
        elif(layer,layers.Dense):
            percent=dense_percent
        if len(weights[idx]) <= 0:
            continue
        #percent = 1.0 - PRUNING_PERCENTAGES[idx]
        rows, cols = weights[idx].shape
        mask[idx] = np.ones((rows, cols))
        k = np.round((rows*cols)*percent).astype(int)
        flat_weights = weights[idx].flatten()
        flat_mask = mask[idx].flatten()
        partition = np.argpartition(np.abs(flat_weights),k)[0:k]
        flat_mask[partition] = 0
        mask[idx] = flat_mask.reshape((rows,cols))
    return mask