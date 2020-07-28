import numpy as np
from framework.losses import Loss

class BCE(Loss):
    @staticmethod
    def calculate(outputs, targets):
        return -np.mean(targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs))
    
    @staticmethod
    def differentiate(outputs, targets):
        return -(targets / outputs) + (1 - targets) / (1 - outputs)