import numpy as np
from neunet import activations as a

np.random.seed(0)

class Dense:
    def __init__(self, n_inputs, n_neurons, activation = None):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
        self.init_activation(activation)

    def init_activation(self, activation):
        if activation == 'sigmoid':
            self.activation = a.sigmoid
        elif activation == 'relu':
            self.activation = a.relu
        elif activation == 'softmax':
            self.activation = a.softmax
        else:
            self.activation = a.linear
        
    def activate(self, inputs):
        self.inputs = inputs
        
        self.outputs = self.activation(self.inputs @ self.weights + self.biases)
        
        return self.outputs
    
    def differentiate(self, inputs):
        return self.activation(inputs, derivative = True)