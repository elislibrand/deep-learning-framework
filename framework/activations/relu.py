import numpy as np
from framework.activations import Activation

class ReLU(Activation):
    @staticmethod
    def activate(z):
        return z * (z > 0)
    
    @staticmethod
    def differentiate(z):
        return 1 * (z > 0)