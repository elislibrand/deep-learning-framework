import numpy as np
from framework.optimizers import Optimizer

class RMSProp(Optimizer):
    def __init__(self, learning_rate = 1e-3, rho = 0.9, epsilon = 1e-8, momentum = 0):
        self.learning_rate = learning_rate
        
        self.rho = rho
        self.epsilon = epsilon
        self.momentum = momentum
        
        self.previous = {}
        
    def optimize(self, layer):
        if not layer in self.previous.keys():
            self.previous[layer] = {
                'weights': np.zeros(layer.weights.shape),
                'biases': np.zeros(layer.biases.shape),
                'rms_weights': np.zeros(layer.weights.shape),
                'rms_biases': np.zeros(layer.biases.shape)
            }
        
        momentum_weights = self.momentum * self.previous.get(layer)['weights']
        momentum_biases = self.momentum * self.previous.get(layer)['biases']
        
        rms_weights = self.rho * self.previous.get(layer)['rms_weights'] + (1 - self.rho) * (layer.gradients_weights ** 2)
        rms_biases = self.rho * self.previous.get(layer)['rms_biases'] + (1 - self.rho) * (layer.gradients_biases ** 2)
        
        adjustment_weights = momentum_weights + (self.learning_rate * layer.gradients_weights) / ((rms_weights ** 0.5) + self.epsilon)
        adjustment_biases = momentum_biases + (self.learning_rate * layer.gradients_biases) / ((rms_biases ** 0.5) + self.epsilon)
        
        layer.weights -= adjustment_weights
        layer.biases -= adjustment_biases
        
        self.previous.get(layer)['weights'] = adjustment_weights
        self.previous.get(layer)['biases'] = adjustment_biases
        self.previous.get(layer)['rms_weights'] = rms_weights
        self.previous.get(layer)['rms_biases'] = rms_biases