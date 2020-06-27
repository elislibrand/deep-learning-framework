import numpy as np
from models import Sequential
from layers import Dense

inputs = np.array([[0, 0, 0],
                   [255, 255, 255],
                   [255, 255, 0],
                   [0, 0, 255],
                   [0, 255, 0]])

processed_inputs = np.array([[0, 0, 0],
                             [1, 1, 1],
                             [1, 1, 0],
                             [0, 0, 1],
                             [0, 1, 0]])

targets = np.array([[1],
                    [0],
                    [0],
                    [1],
                    [0]])

model = Sequential()

model.add(Dense(n_inputs = 3, n_neurons = 3, activation = 'sigmoid'))
model.add(Dense(n_inputs = 3, n_neurons = 1, activation = 'sigmoid'))

model.fit(processed_inputs, targets, eta = 0.05, n_epochs = 50000)