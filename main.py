import numpy as np
from models import Sequential
from layers import Dense

inputs = [[0, 0, 0],
          [255, 255, 255],
          [255, 255, 0],
          [0, 0, 255],
          [0, 255, 0]]

processed_inputs = [[0, 0, 0],
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 1],
                    [0, 1, 0]]

targets = [[1],
           [0],
           [0],
           [1],
           [0]]

model = Sequential()

model.add(Dense(n_inputs = 3, n_neurons = 3, activation = 'relu'))
model.add(Dense(n_inputs = 3, n_neurons = 1, activation = 'sigmoid'))

model.fit(processed_inputs, targets)