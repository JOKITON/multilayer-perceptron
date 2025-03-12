import config
from config import N_LAYERS
from activations import sigmoid, relu, der_sigmoid, der_relu, leaky_relu, der_leaky_relu, tanh, der_tanh, softmax, der_softmax
from dense import DenseLayer

def setup_layers(f_actv, f_der_actv, layer_shape, seed, velocity=False):
    layers = []
    for layer in range(N_LAYERS):
        prev_neurons = layer_shape[layer][0]
        out_neurons = layer_shape[layer][1]
        if (layer != (N_LAYERS - 1)):
            layers.append(DenseLayer(prev_neurons, out_neurons, leaky_relu, der_leaky_relu, seed, velocity))
        else:
            layers.append(DenseLayer(prev_neurons, out_neurons, f_actv, f_der_actv, seed, velocity))
        # print(f"Layer {layer} - {prev_neurons} -> {out_neurons}")
    return layers
