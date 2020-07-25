import numpy as np
from framework.regularizers import Regularizer

class L2(Regularizer):
    def __init__(self, amount = 1e-2):
        self.amount = amount
        
    def regularize(self, layer):
        layer.gradients += 2 * layer.weights * self.amount