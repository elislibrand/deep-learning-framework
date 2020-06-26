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
                layer.error = targets - outputs
            else:
                next_layer = self.layers[i + 1]
                
                layer.error = np.dot(next_layer.weights, next_layer.delta.T).T
                
            layer.delta = layer.error * layer.derivate(layer.outputs)
            
            print(layer.delta)
            print()
                
    def predict(self, inputs):
        predictions = self.forward(inputs)
        
        return np.round(predictions)
    
    def fit(self, inputs, targets, n_epochs: int = 100):
        self.inputs = inputs
        self.targets = targets
        
        outputs = self.forward(self.inputs)
        self.backward(self.targets, outputs)
        
        #print(self.targets - self.outputs)