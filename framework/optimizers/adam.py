import numpy as np
from framework.optimizers import Optimizer

class Adam(Optimizer):
    def __init__(self, learning_rate = 1e-3, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.learning_rate = learning_rate
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.timestep = 0
        
        self.previous = {}
    
    def optimize(self, layer):
        if not layer in self.previous.keys():
            self.previous[layer] = {
                'first_moment_weights': np.zeros(layer.weights.shape),
                'first_moment_biases': np.zeros(layer.biases.shape),
                'second_moment_weights': np.zeros(layer.weights.shape),
                'second_moment_biases': np.zeros(layer.biases.shape)
            }
        
        self.timestep += 1
        
        first_moment_weights = self.beta1 * self.previous.get(layer)['first_moment_weights'] + (1 - self.beta1) * layer.gradients_weights
        first_moment_biases = self.beta1 * self.previous.get(layer)['first_moment_biases'] + (1 - self.beta1) * layer.gradients_biases
        second_moment_weights = self.beta2 * self.previous.get(layer)['second_moment_weights'] + (1 - self.beta2) * (layer.gradients_weights ** 2)
        second_moment_biases = self.beta2 * self.previous.get(layer)['second_moment_biases'] + (1 - self.beta2) * (layer.gradients_biases ** 2)
        
        corrected_first_moment_weights = first_moment_weights / (1 - (self.beta1 ** self.timestep))
        corrected_first_moment_biases = first_moment_biases / (1 - (self.beta1 ** self.timestep))
        corrected_second_moment_weights = second_moment_weights / (1 - (self.beta2 ** self.timestep))
        corrected_second_moment_biases = second_moment_biases / (1 - (self.beta2 ** self.timestep))
        
        adjustment_weights = self.learning_rate * corrected_first_moment_weights / ((corrected_second_moment_weights ** 0.5) + self.epsilon)
        adjustment_biases = self.learning_rate * corrected_first_moment_biases / ((corrected_second_moment_biases ** 0.5) + self.epsilon)
        
        layer.weights -= adjustment_weights
        layer.biases -= adjustment_biases
        
        self.previous.get(layer)['first_moment_weights'] = first_moment_weights
        self.previous.get(layer)['first_moment_biases'] = first_moment_biases
        self.previous.get(layer)['second_moment_weights'] = second_moment_weights
        self.previous.get(layer)['second_moment_biases'] = second_moment_biases