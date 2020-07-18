import numpy as np
from framework import activations

np.random.seed(0)

class Dense:
    def __init__(self, n_inputs, n_neurons, activation = None, regularizers = []):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
        self.activation = activations.get(activation)
        
        self.regularizers = regularizers
        
    def activate(self, inputs):
        self.inputs = inputs
        
        self.outputs = self.activation(self.inputs @ self.weights + self.biases)
        
        return self.outputs
    
    def differentiate(self, inputs):
        return self.activation(inputs, derivative = True)
    
    def regularize(self, eta):
        for regularizer in self.regularizers:
            self.weights = regularizer.update(self.weights, eta)