import numpy as np

class MaxNorm:
    def __init__(self, limit = 2):
        self.limit = limit
        
    def update(self, weights, eta):
        values = np.array(weights)
        
        values[np.where(values > self.limit)] = self.limit
        values[np.where(values < -self.limit)] = -self.limit
        
        return values