import numpy as np

class HeUniform:
    @staticmethod
    def initialize(n_inputs, n_neurons):
        return np.random.uniform(
            low = -np.sqrt(6 / n_inputs),
            high = np.sqrt(6 / n_inputs),
            size = (n_inputs, n_neurons)
        )