import numpy as np
from activations import sigmoid, relu
from config import LEARNING_RATE, N_LAYERS, LAMBDA_REG

def stepforward(_X, _weights, _bias):
    """ 
        Apply forward propagation to get predictions.
        
        Parameters:
            - _X: Feature matrix, shape (n_samples, n_features)
            - _weights: Weights, shape (n_features,)
            - _bias: Bias, shape (1,)
            
        Returns:
            - predictions: Predicted target values, shape (n_samples,)
    """
    z = np.dot(_X, _weights) + _bias
    # a = sigmoid(z)  # Apply sigmoid activation
    return z

def forward_propagation(X, weights, biases, activation_function=sigmoid):
    """
    Perform a single forward pass through the network.

    Parameters:
        - X: Input data, shape (n_samples, n_features)
        - weights: List of weights for each layer
        - biases: List of biases for each layer
        - activation_function: Activation function to apply at each layer (default: sigmoid)

    Returns:
        - activations: List of activations for each layer, including the input layer
        - z_values: List of z values (pre-activation values) for each layer
    """
    activations = [X]  # Start with input features
    z_values = []      # To store z = Wx + b for each layer

    for i in range(len(weights)):  # Iterate over layers
        z = stepforward(activations[-1], weights[i], biases[i])  # Compute z
        z_values.append(z)  # Save z for backpropagation
        if (i == len(weights) - 1):
            a = activation_function(z)
        else:
            a = relu(z)  # Apply activation
        activations.append(a)  # Save activation for next layer

    return activations, z_values

def sigmoid_derivative(x):
    """
    Compute the derivative of the sigmoid function.
    
    Parameters:
        - x: Input to the sigmoid function.
    
    Returns:
        - Derivative of the sigmoid function.
    """
    return sigmoid(x) * (1 - sigmoid(x))

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def compute_gradients(activations, z_values, y_train, weights, biases, activation_derivative=relu_derivative):
    """
    Compute gradients of weights and biases for a neural network using backpropagation.
    
    Parameters:
        - activations: List of activations from each layer, including input (size: N_LAYERS + 1).
        - z_values: List of z-values (pre-activation values) for each layer.
        - y_train: Ground truth labels, shape (n_samples, output_dim).
        - weights: List of weight matrices for each layer (size: N_LAYERS).
        - biases: List of bias vectors for each layer (size: N_LAYERS).
        - activation_derivative: Derivative of the activation function.

    Returns:
        - weight_gradients: List of gradients for weights (size: N_LAYERS).
        - bias_gradients: List of gradients for biases (size: N_LAYERS).
    """
    # Initialize lists to store gradients
    weight_gradients = [None] * N_LAYERS
    bias_gradients = [None] * N_LAYERS
    
    # Compute output layer error (delta)
    delta = activations[-1] - y_train  # Derivative of MSE Loss: (y_pred - y_true)

    # Backpropagate through layers
    for l in reversed(range(N_LAYERS)):
        # Gradients for weights and biases
        weight_gradients[l] = np.dot(activations[l].T, delta) / y_train.shape[0]  # Average over samples
        weight_gradients[l] += (LAMBDA_REG / y_train.shape[0]) * weights[l]

        bias_gradients[l] = np.sum(delta, axis=0, keepdims=True) / y_train.shape[0]

        # Compute delta for the next layer (if not the input layer)
        if l > 0:
            delta = np.dot(delta, weights[l].T) * activation_derivative(z_values[l - 1])
    
    return weight_gradients, bias_gradients
