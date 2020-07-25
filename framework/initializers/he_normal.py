import numpy as np
from framework.initializers import Initializer

class HeNormal(Initializer):
    @staticmethod
    def initialize(n_inputs, n_neurons):
        return np.random.normal(
            scale = np.sqrt(2 / n_inputs),
            size = (n_inputs, n_neurons)
        )