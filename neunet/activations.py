import numpy as np

def linear(z, derivative = False):
    if derivative:
        return 1
    
    return z

def sigmoid(z, derivative = False):
    if derivative:
        return z * (1 - z)
    
    return 1 / (1 + np.exp(-z))

def relu(z, derivative = False):
    if derivative:
        return 1 * (z > 0)
    
    return z * (z > 0)

def softmax(z, derivative = False):
    if derivative:
        pass
    
    return np.exp(z - np.max(z)) / np.sum(np.exp(z - np.max(z)))