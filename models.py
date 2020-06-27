import numpy as np
from layers import Dense

class Sequential:
    def __init__(self):
        self.layers = []
        
    def add(self, layer: Dense):
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
                
                layer.errors = np.dot(next_layer.deltas, next_layer.weights.T)
                
            layer.deltas = layer.errors * layer.derivate(layer.outputs)
            
            layer.weights += self.eta * np.dot(layer.inputs.T, layer.deltas)
            layer.biases += self.eta * np.sum(layer.deltas, axis = 0, keepdims = True)
            
            #self.print_matrices(i)
            
    def print_matrices(self, i):
        print('Layer {}'.format(i))
        
        print('Inputs shape: {}'.format(self.layers[i].inputs.shape))
        print('Weights shape: {}'.format(self.layers[i].weights.shape))
        print('Biases shape: {}'.format(self.layers[i].biases.shape))
        print('Errors shape: {}'.format(self.layers[i].errors.shape))
        print('Derivatives shape: {}'.format(self.layers[i].derivate(self.layers[i].outputs).shape))
        print('Deltas shape: {}'.format(self.layers[i].deltas.shape))
        print()
            
    def predict(self, inputs):
        predictions = self.forward(inputs)
        
        return np.round(predictions)
    
    def fit(self, inputs, targets, eta: float, n_epochs: int = 10000):
        self.inputs = inputs
        self.targets = targets
        self.eta = eta
        
        for i in range(n_epochs):
            outputs = self.forward(self.inputs)
            self.backward(self.targets, outputs)
            
            if i % (n_epochs / (n_epochs / 1000)) == 0:
                print('Epoch {}\n{}\n'.format(i, self.forward(self.inputs)))