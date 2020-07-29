import numpy as np
from framework.optimizers import Optimizer

class Adam(Optimizer):
    def __init__(self, learning_rate = 1e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
        self.learning_rate = learning_rate
        
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
                
        self.previous = {}
    
    def optimize(self, layer):
        if not layer in self.previous.keys():
            self.previous[layer] = {
                'beta_1': 1,
                'beta_2': 1,
                'first_moment_weights': np.zeros(layer.weights.shape),
                'first_moment_biases': np.zeros(layer.biases.shape),
                'second_moment_weights': np.zeros(layer.weights.shape),
                'second_moment_biases': np.zeros(layer.biases.shape)
            }
                
        first_moment_weights = self.beta_1 * self.previous.get(layer)['first_moment_weights'] + (1 - self.beta_1) * layer.gradients_weights
        first_moment_biases = self.beta_1 * self.previous.get(layer)['first_moment_biases'] + (1 - self.beta_1) * layer.gradients_biases
        second_moment_weights = self.beta_2 * self.previous.get(layer)['second_moment_weights'] + (1 - self.beta_2) * (layer.gradients_weights ** 2)
        second_moment_biases = self.beta_2 * self.previous.get(layer)['second_moment_biases'] + (1 - self.beta_2) * (layer.gradients_biases ** 2)
        
        beta_1_step = self.beta_1 * self.previous.get(layer)['beta_1']
        beta_2_step = self.beta_2 * self.previous.get(layer)['beta_2']
        
        corrected_first_moment_weights = first_moment_weights / (1 - beta_1_step)
        corrected_first_moment_biases = first_moment_biases / (1 - beta_1_step)
        corrected_second_moment_weights = second_moment_weights / (1 - beta_2_step)
        corrected_second_moment_biases = second_moment_biases / (1 - beta_2_step)
        
        adjustment_weights = self.learning_rate * corrected_first_moment_weights / ((corrected_second_moment_weights ** 0.5) + self.epsilon)
        adjustment_biases = self.learning_rate * corrected_first_moment_biases / ((corrected_second_moment_biases ** 0.5) + self.epsilon)
        
        layer.weights -= adjustment_weights
        layer.biases -= adjustment_biases
        
        self.previous.get(layer)['beta_1'] = beta_1_step
        self.previous.get(layer)['beta_2'] = beta_2_step
        self.previous.get(layer)['first_moment_weights'] = first_moment_weights
        self.previous.get(layer)['first_moment_biases'] = first_moment_biases
        self.previous.get(layer)['second_moment_weights'] = second_moment_weights
        self.previous.get(layer)['second_moment_biases'] = second_moment_biases