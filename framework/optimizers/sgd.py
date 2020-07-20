import numpy as np

class SGD:
    def __init__(self, learning_rate = 0.01, momentum = 0):
        self.learning_rate = learning_rate
        
        self.momentum = momentum
        
    def update(self, layer):
        momentum_weights = self.momentum * layer.previous_adjustment_weights
        momentum_biases = self.momentum * layer.previous_adjustment_biases
        
        adjustment_weights = self.learning_rate * layer.gradients + momentum_weights
        adjustment_biases = self.learning_rate * np.sum(layer.deltas, axis = 0, keepdims = True) + momentum_biases
        
        layer.weights += adjustment_weights
        layer.biases += adjustment_biases
        
        layer.previous_adjustment_weights = adjustment_weights
        layer.previous_adjustment_biases = adjustment_biases