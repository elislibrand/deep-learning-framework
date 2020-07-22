import numpy as np
from framework.regularizers import Regularizer

class L1(Regularizer):
    def __init__(self, amount = 1e-2):
        self.amount = amount
        
    def regularize(self, layer):
        layer.gradients -= np.sign(layer.weights) * self.amount