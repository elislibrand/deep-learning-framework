import numpy as np
from layers import Dense

class Sequential:
    def __init__(self):
        self.layers = []
        
    def add(self, layer: Dense):
        self.layers.append(layer)
        
    def forward(self, inputs):
        pass