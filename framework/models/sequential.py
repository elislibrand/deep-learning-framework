import numpy as np

class Sequential:
    def __init__(self, layers = []):
        self.layers = layers
        
    def add(self, layer):
        self.layers.append(layer)
        
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
            
            layer.weights += self.eta * layer.inputs.T @ layer.deltas
            layer.biases += self.eta * np.sum(layer.deltas, axis = 0, keepdims = True)
            
            layer.regularize(self.eta)
            
    def predict(self, inputs):
        predictions = self.forward(inputs)
        
        return np.around(predictions)
    
    def fit(self, inputs, targets, eta, n_epochs):
        self.eta = eta
        
        for i in range(n_epochs):
            outputs = self.forward(inputs)
            self.backward(targets, outputs)
            
            if i % (n_epochs / (n_epochs / 1000)) == 0:
                print('Epoch {}\t{}'.format(i, np.around(self.forward(inputs).T, decimals = 3)), end = '\n')