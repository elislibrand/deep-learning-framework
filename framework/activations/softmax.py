import numpy as np
from framework.activations import Activation

class Softmax(Activation):
    @staticmethod
    def activate(z):
        return np.exp(z - np.max(z)) / np.sum(np.exp(z - np.max(z)))