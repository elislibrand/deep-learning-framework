import numpy as np

class Dropout:
    def __init__(self, n_neurons, rate):
        self.n_neurons = n_neurons

        self.rate = rate
        
    def activate(self, inputs):
        self.inputs = inputs
        
        self.mask = np.zeros(self.n_neurons, dtype = bool)
        self.mask[np.where(np.random.uniform(size = self.n_neurons) < self.rate)] = True

        self.outputs = self.inputs
        
        self.outputs[:, self.mask] = 0
        self.outputs[:, np.logical_not(self.mask)] *= 1 / (1 - self.rate)
        
        return self.outputs
    
    def differentiate(self, inputs):
        pass
    
    def regularize(self, eta):
        pass