import numpy as np
from framework.initializers import Initializer

class GlorotUniform(Initializer):
    @staticmethod
    def initialize(n_inputs, n_neurons):        
        return np.random.uniform(
            low = -np.sqrt(6 / (n_inputs + n_neurons)),
            high = np.sqrt(6 / (n_inputs + n_neurons)),
            size = (n_inputs, n_neurons)
        )