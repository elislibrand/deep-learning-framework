import numpy as np
from framework.optimizers import Optimizer

class RMSProp(Optimizer):
    def __init__(self, learning_rate = 1e-3, rho = 0.9, epsilon = 1e-10, momentum = 0):
        self.learning_rate = learning_rate
        
        self.rho = rho
        self.epsilon = epsilon
        self.momentum = momentum
        
        self.previous = {}
        
    def optimize(self, layer):
        if not layer in self.previous.keys():
            self.previous[layer] = {'weights': np.zeros(layer.weights.shape),
                                    'biases': np.zeros(layer.biases.shape),
                                    'rms_weights': np.zeros(layer.weights.shape),
                                    'rms_biases': np.zeros(layer.biases.shape)}
        
        momentum_weights = self.momentum * self.previous.get(layer)['weights']
        momentum_biases = self.momentum * self.previous.get(layer)['biases']
        
        rms_weights = self.rho * self.previous.get(layer)['rms_weights'] + (1 - self.rho) * (layer.gradients ** 2)
        rms_biases = self.rho * self.previous.get(layer)['rms_biases'] + (1 - self.rho) * (np.sum(layer.deltas, axis = 0, keepdims = True) ** 2)
        
        adjustment_weights = momentum_weights + layer.gradients / np.sqrt(rms_weights + self.epsilon)
        adjustment_biases = momentum_biases + np.sum(layer.deltas, axis = 0, keepdims = True) / np.sqrt(rms_biases + self.epsilon)
        
        layer.weights -= self.learning_rate * adjustment_weights
        layer.biases -= self.learning_rate * adjustment_biases
        
        self.previous.get(layer)['weights'] = adjustment_weights
        self.previous.get(layer)['biases'] = adjustment_biases
        self.previous.get(layer)['rms_weights'] = rms_weights
        self.previous.get(layer)['rms_biases'] = rms_biases