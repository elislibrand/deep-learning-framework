import numpy as np

def linear(z, derivative: bool = False):
    if derivative:
        return 1
    
    return z

def sigmoid(z, derivative: bool = False):
    if derivative:
        return z * (1 - z)
    
    return 1 / (1 + np.exp(-z))

def relu(z, derivative: bool = False):
    if derivative:
        return np.heaviside(z, 0)
    
    return np.maximum(0, z)

def softmax(z, derivative: bool = False):
    if derivative:
        pass
    
    return np.exp(z - np.max(z)) / np.sum(np.exp(z - np.max(z)))