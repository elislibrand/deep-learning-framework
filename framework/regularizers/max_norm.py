import numpy as np
from framework.regularizers import Regularizer

class MaxNorm(Regularizer):
    def __init__(self, limit = 2):
        self.limit = limit
        
    def update_weights(self, layer):
        layer.weights = np.minimum(self.limit, layer.weights)
        layer.weights = np.maximum(-self.limit, layer.weights)