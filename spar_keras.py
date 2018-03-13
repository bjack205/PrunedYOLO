from keras import backend as K
import keras.layers
from keras.models import Model
import numpy as np

def get_prunable_layers(model):
    prunable_layers = []
    for layer in model.layers:
        if layer.trainable and layer.name.startswith('conv2d'):
            prunable_layers.append(layer)
    return prunable_layers

def calc_cutoff(prunable_layers):
    data = []
    for layer in prunable_layers:
        for w in layer.get_weights():
            data.extend(np.abs(w.flatten()))
    print("Number of Params:", len(data))
    return np.percentile(data, 50)


def prune_weights(prunable_layers, cutoff):
    num_pruned = []
    for layer in prunable_layers:
        weights = layer.get_weights()
        n_prune = 0
        for w in weights:
            mask = w > cutoff
            w *= mask
            n_prune += np.sum(mask)
        layer.set_weights(weights)
    return num_pruned




