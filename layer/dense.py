""" File to create the DenseLayer class. """

import numpy as np
from activations import sigmoid, relu, der_sigmoid, der_relu, leaky_relu, der_leaky_relu, softmax

class DenseLayer:
    """ DenseLayer class used for creating a dense layer in a neural network. """
    # General Use
    out_layer = False
    # Forward Propagation
    inputs = None
    weights = None
    biases = None
    output = None
    activations = None
    # Backwward Propagation
    dinputs = None
    nesterov = False

    def __init__(self, n_inputs, n_neurons, actv, der_actv, seed=None, velocity=False):
        """
        Initialize the Dense layer.
        
        Parameters:
        n_inputs  - Number of inputs to the layer.
        n_neurons - Number of neurons in the layer.
        """
        # Weight initialization
        if n_neurons == 1:  # Output layer (Sigmoid)
            scale = np.sqrt(1 / n_inputs)
        else:  # Hidden layers (Leaky ReLU)
            scale = np.sqrt(2 / n_inputs)
        if seed is not None:
            np.random.seed(seed)  # Set the seed for reproducibility

        self.weights = np.random.randn(n_inputs, n_neurons) * scale
        self.biases = np.zeros((1, n_neurons))

        self.f_actv = actv
        self.f_der_actv = der_actv
        
        if (velocity == True):
            self.velocity = np.zeros_like(self.weights)
        else:
            self.velocity = None

    def forward(self, inputs=None):
        """ 
		Perform forward propagation for the layer
  
		Parameters:
		inputs	- Input data

		Returns:
		Output of the layer after forward propagation.
        """
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.biases
        """ print("Forward:")
        print(self.inputs.shape, self.weights.shape, self.biases.shape)
        print(self.output.shape)
        print() """

        self.activations = self.f_actv(self.output)

        return self.activations, self.output

    def backward(self, yvalues, learning_rate, momentum=0):
        """ 
		Perform backward propagation for the layer
  
		Parameters:
		dvalues			- Gradient of the loss with respect to the output.
		learning_rate	- The rate for gradient descent
        """
        # Partial derivative of BCE loss with respect to activations
        dactivation = self.activations - yvalues

        # Gradients with respect to weights, biases, and inputs
        dweights = np.dot(self.inputs.T, dactivation) / self.activations.shape[0]
        """ print("Backward:")
        print(dactivation.shape, self.inputs.shape)
        print(dactivation.shape, self.inputs.T.shape)
        print(dweights.shape)
        print() """
        dbiases = np.sum(dactivation, axis=0, keepdims=True) / self.activations.shape[0]

        # Gradient with respect to inputs
        self.dinputs = np.dot(dactivation, self.weights.T)

        # Update biases (biases usually don't use momentum)
        self.biases -= learning_rate * dbiases

        # Initialize velocity if it's not provided (first call)
        if self.velocity is None:
            self.weights -= learning_rate * dweights
        else: # Apply Nesterov Momentum
            # Look ahead by applying momentum
            lookahead_weights = self.weights + momentum * self.velocity

            # Recompute gradients at the "look-ahead" weights
            dweights = np.dot(self.inputs.T, dactivation) / self.activations.shape[0]

            # Clip gradients
            np.clip(dweights, -1, 1, out=dweights)
            np.clip(dbiases, -1, 1, out=dbiases)

            # Update velocity and weights
            self.velocity = (momentum * self.velocity) - (learning_rate * dweights)
            self.weights += self.velocity

            # Return updated velocity to maintain state across iterations
            return self.velocity
