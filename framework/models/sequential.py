import numpy as np
from framework.optimizers import SGD

class Sequential:
    def __init__(self, layers = []):
        self.layers = layers
        
        self.is_built = False
        
    def add(self, layer):
        self.layers.append(layer)
        
    def build(self, n_inputs, seed = None):
        if self.is_built:
            return
        
        for i in range(len(self.layers)):
            layer = self.layers[i]
            
            if layer == self.layers[0]:
                layer.build(n_inputs = n_inputs, seed = seed)
            else:
                previous_layer = self.layers[i - 1]
                
                layer.build(n_inputs = previous_layer.n_neurons, seed = seed)
                
        self.is_built = True
        
    def compile(self, optimizer):
        self.optimizer = optimizer
        
    def forward(self, inputs):
        X = inputs
        
        for layer in self.layers:
            X = layer.activate(X)

        return X
    
    def backward(self, targets, outputs):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            
            if layer == self.layers[-1]:
                layer.errors = targets - outputs
            else:
                next_layer = self.layers[i + 1]
                
                layer.errors = next_layer.deltas @ next_layer.weights.T
                
            layer.deltas = layer.errors * layer.differentiate(layer.outputs)
            layer.gradients = layer.inputs.T @ layer.deltas
            
            layer.regularize_gradients()
            
            self.optimizer.update(layer)
            
            layer.regularize_weights()
            
    def predict(self, inputs):
        predictions = self.forward(inputs)
        
        return np.around(predictions)
    
    def shuffle(self, inputs, targets):
        indices = np.random.permutation(len(targets))
        
        return inputs[indices], targets[indices]
    
    def fit(self, inputs, targets, batch_size, n_epochs, shuffle = True, seed = None):
        self.build(inputs.shape[1], seed)
        
        original_inputs = np.array(inputs)
                
        for i in range(n_epochs):
            if shuffle:
                inputs, targets = self.shuffle(inputs, targets)
            
            for j in range(0, len(targets), batch_size):
                inputs_batch, targets_batch = inputs[j:j + batch_size], targets[j:j + batch_size]

                outputs_batch = self.forward(inputs_batch)
                self.backward(targets_batch, outputs_batch)

            print('Epoch {}\t{}'.format(i, np.around(self.forward(original_inputs).T, decimals = 3)), end = '\n')