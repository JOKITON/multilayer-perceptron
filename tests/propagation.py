import numpy as np
from activations import sigmoid

LEARNING_RATE = 0.1
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
    a = sigmoid(z)  # Apply sigmoid activation

    return a, z

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))  # Sigmoid derivative

def compute_gradient(_X, _y_train, _activated, _weights, _biases):
    """
    Computes the gradient of the weights and biases.
    Update values accordingly using the given learning rate.
    
    Parameters:
        - _X: Feature matrix, shape (n_samples, n_features)
        - _y_train: Target values, shape (n_samples, 1)
        - _predicted: Predicted target values, shape (n_samples,)
        - _weights: Current weights, shape (1, n_features)
        - _biases: Current biases, shape (1,)

    Returns:
        - new_weights: Updated weights, shape (n_features,)
        - new_biases: Updated biases, shape (1,)
    """
    n_samples = len(_activated)

    # Compute the error (difference between prediction and true value)
    # (455,) - (455, 1) 
    error = _activated - _y_train.T
    error = error.flatten()

    # Backpropagate the error through the sigmoid derivative
    delta = error * _activated * (1 - _activated)  # Element-wise product

    # Compute gradients for weights and bias
    dW = np.dot(_X.T, delta) / n_samples  # Shape: (n_features,)
    db = np.sum(delta) / n_samples  # Scalar
    # print("dW:", dW, "db:", db)

    # Update weights and biases
    new_weights = _weights - LEARNING_RATE * dW
    new_biases = _biases - LEARNING_RATE * db

    return new_weights, new_biases