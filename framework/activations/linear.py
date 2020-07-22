import numpy as np
from framework.activations import Activation

class Linear(Activation):
    @staticmethod
    def activate(z):
        return z
    
    @staticmethod
    def differentiate(z):
        return 1