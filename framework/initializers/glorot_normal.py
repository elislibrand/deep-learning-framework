import numpy as np
from framework.initializers import Initializer
class GlorotNormal(Initializer):
    @staticmethod
    def initialize(n_inputs, n_neurons):        
        return np.random.normal(
            scale = np.sqrt(2 / (n_inputs + n_neurons)),
            size = (n_inputs, n_neurons)
        )