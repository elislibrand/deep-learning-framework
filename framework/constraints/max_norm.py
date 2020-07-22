import numpy as np
from framework.constraints import Constraint

class MaxNorm(Constraint):
    def __init__(self, limit = 2):
        self.limit = limit
        
    def constrain(self, layer):
        layer.weights = np.minimum(self.limit, layer.weights)
        layer.weights = np.maximum(-self.limit, layer.weights)