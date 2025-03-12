""" Activation functions for neural networks. """

import numpy as np

def sigmoid(x):
    """
    Activates the values in the range (0, 1)
    
    Paremeters:
        - x: Values to activate
        
    Returns:
        - Array of the activated values
    """
    return 1 / (1 + np.exp(-x))

def relu(x):
    """
    Activates the values in the range max(0, x)
    
    Paremeters:
        - x: Values to activate
        
    Returns:
        - Array of the activated values
    """
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def der_softmax(z, y_true):
    """Compute the gradient of the softmax function combined with cross-entropy loss.
    
    Parameters:
    z       - Logits (raw scores), shape (n_samples, n_classes)
    y_true  - True labels (one-hot encoded), shape (n_samples, n_classes)
    
    Returns:
    gradient - Gradient of the loss with respect to the logits, shape (n_samples, n_classes)
    """
    # Compute softmax probabilities
    y_pred = softmax(z)
    # Gradient: softmax output - true labels
    gradient = y_pred - y_true
    return gradient

def der_leaky_relu(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def tanh(x):
    """
    Activates the values in the range (-1, 1)
    
    Paremeters:
        - x: Values to activate
        
    Returns:
        - Array of the activated values
    """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def der_sigmoid(x):
    """
    Compute the derivative of the Sigmoid activation function.
    
    Parameters:
        - x: Input to the sigmoid function.
    
    Returns:
        - Derivative of the sigmoid function.
    """
    return sigmoid(x) * (1 - sigmoid(x))

def der_relu(x):
    """ 
    Compute the derivative of the RELU activation function.

    Parameters:
        - x: Input to the relu function.

    Returns:
        - Derivative of the relu function.
    """
    return np.where(x > 0, 1, 0)

def der_tanh(x):
    """ 
    Compute the derivative of the tanh activation function.

    Parameters:
        - x: Input to the tanh function (scalar or np.ndarray).

    Returns:
        - Derivative of the tanh function.
    """
    tanh_x = np.tanh(x)
    return 1 - tanh_x**2
