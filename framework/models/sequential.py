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
        
    def compile(self, optimizer = SGD()):
        self.optimizer = optimizer
        
    @staticmethod
    def shuffle(inputs, targets):
        indices = np.random.permutation(len(targets))
        
        return inputs[indices], targets[indices]
        
    @staticmethod
    def get_accuracy(targets, outputs):
        return np.mean(targets == np.around(outputs))
        
    def forward(self, inputs):
        X = inputs
        
        for layer in self.layers:
            X = layer.activate(X)

        return X
    
    def backward(self, targets, outputs):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            
            if layer == self.layers[-1]:
                layer.errors = targets - outputs # Change to: outputs - targets
            else:
                next_layer = self.layers[i + 1]
                
                layer.errors = next_layer.deltas @ next_layer.weights.T
                
            layer.deltas = layer.errors * layer.differentiate(layer.outputs)
            layer.gradients = layer.inputs.T @ layer.deltas
            
            layer.regularize()
            
            self.optimizer.optimize(layer)
            
            layer.constrain()
    
    def fit(self, inputs, targets, batch_size, n_epochs, shuffle = True, seed = None):
        self.build(inputs.shape[1], seed)
                        
        for i in range(n_epochs):
            if shuffle:
                inputs, targets = self.shuffle(inputs, targets)

            for j in range(0, len(targets), batch_size):
                inputs_batch, targets_batch = inputs[j:j + batch_size], targets[j:j + batch_size]

                outputs_batch = self.forward(inputs_batch)
                self.backward(targets_batch, outputs_batch)
                        
            print('Epoch {:<{width}}    [accuracy: {:.3f} - loss: {:.3f}]'.format((i + 1),
                                                                   np.around(self.get_accuracy(targets, self.forward(inputs)), 3),
                                                                   np.around(np.mean(-(targets * np.log(self.forward(inputs)) + (1 - targets) * np.log(1 - self.forward(inputs)))), 3),
                                                                   width = len(str(n_epochs))))
            
    def predict(self, inputs):
        predictions = self.forward(inputs)
        
        return np.around(predictions)