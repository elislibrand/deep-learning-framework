import numpy as np
from framework.activations import Linear
from framework.initializers import GlorotUniform

np.random.seed(0)

class Dense:
    def __init__(self, n_neurons, activation = Linear(), initializer = GlorotUniform(), regularizers = [], constraints = []):
        self.n_neurons = n_neurons
        
        self.activation = activation
        
        self.initializer = initializer
        
        self.regularizers = regularizers
        self.constraints = constraints
    
    def build(self, n_inputs, seed):
        self.n_inputs = n_inputs
        
        #np.random.seed(seed)
        
        self.weights = self.initializer.initialize(self.n_inputs, self.n_neurons)
        self.biases = np.zeros((1, self.n_neurons))

        #self.previous_adjustment_weights = np.zeros(self.weights.shape)
        #self.previous_adjustment_biases = np.zeros(self.biases.shape)
        
    def activate(self, inputs):
        self.inputs = inputs
        
        self.outputs = self.activation.activate(self.inputs @ self.weights + self.biases)
        
        return self.outputs
    
    def differentiate(self, inputs):
        return self.activation.differentiate(inputs)
    
    def regularize(self):
        for regularizer in self.regularizers:
            regularizer.regularize(self)
            
    def constrain(self):
        for constraint in self.constraints:
            constraint.constrain(self)