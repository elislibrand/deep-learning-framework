import numpy as np
from framework.activations import Activation

class Sigmoid(Activation):
    @staticmethod
    def activate(z):
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def differentiate(z):
        return z * (1 - z)