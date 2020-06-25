import numpy as np
import activations as a

np.random.seed(0)

class Dense:
    def __init__(self, n_inputs: int, n_neurons: int, activation: str = None):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
        self.activation = activation
        
    def forward(self, inputs):
        self.outputs = self.activate(np.dot(inputs, self.weights) + self.biases)
        
    def activate(self, z):
        if self.activation == 'sigmoid':
            outputs = a.sigmoid(z)
        elif self.activation == 'relu':
            outputs = a.relu(z)
        elif self.activation == 'softmax':
            outputs = a.softmax(z)
        else:
            outputs = z    
        
        return outputs