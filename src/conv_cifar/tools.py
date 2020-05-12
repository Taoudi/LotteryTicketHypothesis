from constants import PRUNING_PERCENTAGES
from tensorflow.keras import layers

def generate_percentages(base_percents, lower_bound):
    percentages = {}
    percents = base_percents
    idx = 0
    while percents[1] >= lower_bound:
        new_percents = []
        for i in range(len(base_percents)):
            new_percents.append((1-PRUNING_PERCENTAGES[i])*percents[i])
        
        percentages[idx] = new_percents
        percents = new_percents
        
        idx += 1
    
    return percentages, len(percentages)


def get_weights(network):
    weights = {}
    for idx, layer in enumerate(network.model.layers):
        #weights[idx] = layer.get_weights()[0]    
        if isinstance(layer,layers.Conv2D) or isinstance(layer,layers.Dense):
            weights[idx] = layer.get_weights()[0]
            continue
        weights[idx] = []
    return weights
