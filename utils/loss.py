""" Loss functions for the project. """

import numpy as np
from config import N_LAYERS

def f_loss(y_true, y_pred):
    """ Computes the loss for the given predictions and results. """
    return np.sum(y_true - y_pred)

def f_mse(y_true, y_pred):
    """ Computes the Mean Squared Error for the given inputs, weights, and bias. """
    # Calculate the MSE
    ret_mse = np.mean((y_pred - y_true) ** 2)
    return ret_mse

def f_mae(y_true, y_pred):
    """ Computes the Mean Absolute Error for the given predictions and results. """
    return np.mean(np.abs(y_pred - y_true))

def f_r2score(y_true, y_pred):
    """ Computes the Mean Absolute Error for the given predictions and results. """
    math1 = np.sum((y_true - y_pred) ** 2)
    math2 = np.sum((y_true - np.mean(y_true)) ** 2)
    result = math1/math2
    result = 1 - result
    return result

def f_cross_entropy(y_true, y_pred):
    """ Cross Entropy activation function. """
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_f_cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-15), axis=1).mean()
