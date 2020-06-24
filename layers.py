import numpy as np
from activations import *

np.random.seed(0)

class Dense:
    def __init__(self, n_inputs: int, n_neurons: int, activation: str = None):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        
        self.weights = np.random.randn(n_inputs, n_neurons) #* np.sqrt(2 / (n_inputs + n_neurons))
        self.biases = np.zeros((1, n_neurons))
        
        self.activation = activation
        
    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases # Activate z