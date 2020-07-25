import numpy as np
from framework.losses import Loss

class BinaryCrossEntropy(Loss):
    @staticmethod
    def calculate(targets, outputs):
        return -np.mean(targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs))
    
    @staticmethod
    def differentiate(targets, outputs):
        return -(targets / outputs) + (1 - targets) / (1 - outputs)