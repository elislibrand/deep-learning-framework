import numpy as np
from framework.losses import Loss

class MSE(Loss):
    @staticmethod
    def calculate(targets, outputs):
        return np.mean((outputs - targets) ** 2)
    
    @staticmethod
    def differentiate(targets, outputs):
        return 2 * (outputs - targets)