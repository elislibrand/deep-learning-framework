import numpy as np

class L1:
    def __init__(self, amount = 1e-2):
        self.amount = amount
        
    def update(self, weights, eta):
        values = np.array(weights)
        delta = self.amount * eta
        
        values[np.where(np.abs(values) < delta)] = 0
        values[np.where(values > 0)] -= delta
        values[np.where(values < 0)] += delta
        
        return values