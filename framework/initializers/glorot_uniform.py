import numpy as np

class GlorotUniform:
    @staticmethod
    def initialize(n_inputs, n_neurons):        
        return np.random.uniform(
            low = -np.sqrt(6 / (n_inputs + n_neurons)),
            high = np.sqrt(6 / (n_inputs + n_neurons)),
            size = (n_inputs, n_neurons)
        )