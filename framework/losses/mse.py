import numpy as np
from framework.losses import Loss

class MSE(Loss):
    @staticmethod
    def calculate(outputs, targets):
        return np.mean((outputs - targets) ** 2)
    
    @staticmethod
    def differentiate(outputs, targets):
        return 2 * (outputs - targets)