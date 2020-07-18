import numpy as np

class L2:
    def __init__(self, amount = 1e-2):
        self.amount = amount
        
    def update(self, weights, eta):
        values = np.array(weights)
        delta = self.amount * eta
        
        values -= 2 * delta * values
        
        return values