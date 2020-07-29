import numpy as np
from framework.optimizers import Optimizer

class SGD(Optimizer):
    def __init__(self, learning_rate = 1e-3, momentum = 0):
        self.learning_rate = learning_rate
        
        self.momentum = momentum
        
        self.previous = {}
        
    def optimize(self, layer):
        if not layer in self.previous.keys():
            self.previous[layer] = {
                'weights': np.zeros(layer.weights.shape),
                'biases': np.zeros(layer.biases.shape)
            }
        
        momentum_weights = self.momentum * self.previous.get(layer)['weights']
        momentum_biases = self.momentum * self.previous.get(layer)['biases']
        
        adjustment_weights = momentum_weights + self.learning_rate * layer.gradients_weights
        adjustment_biases = momentum_biases + self.learning_rate * layer.gradients_biases
        
        layer.weights -= adjustment_weights
        layer.biases -= adjustment_biases
        
        self.previous.get(layer)['weights'] = adjustment_weights
        self.previous.get(layer)['biases'] = adjustment_biases