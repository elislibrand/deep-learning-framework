import numpy as np

class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        
    def update(self, layer):
        layer.weights += self.learning_rate * layer.gradients
        layer.biases += self.learning_rate * np.sum(layer.deltas, axis = 0, keepdims = True)