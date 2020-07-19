import numpy as np
from framework import activations

class Dense:
    def __init__(self, n_neurons, activation = None, regularizers = []):
        self.n_neurons = n_neurons
        
        self.activation = activations.get(activation)
        
        self.regularizers = regularizers
    
    def build(self, n_inputs, seed):
        self.n_inputs = n_inputs
        
        np.random.seed(seed)
        
        self.weights = np.random.randn(self.n_inputs, self.n_neurons)
        self.biases = np.zeros((1, self.n_neurons))
    
    def activate(self, inputs):
        self.inputs = inputs
        
        self.outputs = self.activation(self.inputs @ self.weights + self.biases)
        
        return self.outputs
    
    def differentiate(self, inputs):
        return self.activation(inputs, derivative = True)
    
    def regularize_gradients(self):
        for regularizer in self.regularizers:
            regularizer.update_gradients(self)
            
    def regularize_weights(self):
        for regularizer in self.regularizers:
            regularizer.update_weights(self)