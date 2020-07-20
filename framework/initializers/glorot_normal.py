import numpy as np

class GlorotNormal:
    @staticmethod
    def initialize(n_inputs, n_neurons):        
        return np.random.normal(
            scale = np.sqrt(2 / (n_inputs + n_neurons)),
            size = (n_inputs, n_neurons)
        )