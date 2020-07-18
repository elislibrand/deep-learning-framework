import numpy as np

# NOT WORKING PROPERLY
class Dropout:
    def __init__(self, n_neurons, rate):
        self.n_neurons = n_neurons

        self.rate = rate
        
    def activate(self, inputs):
        self.inputs = inputs
        
        self.weights = np.ones((self.n_neurons, self.n_neurons))
        self.biases = np.zeros((self.n_neurons, self.n_neurons))
        
        self.dropout = np.zeros(self.n_neurons, dtype = bool)
        self.dropout[np.where(np.random.uniform(size = self.n_neurons) < self.rate)] = True

        self.outputs = self.inputs
        
        self.outputs[:, self.dropout] = 0
        self.outputs[:, np.logical_not(self.dropout)] *= 1 / (1 - self.rate)
        
        return self.outputs
    
    def differentiate(self, inputs):
        inputs[np.where(inputs > 0)] = 1
        
        return inputs
    
    def regularize(self, eta):
        pass