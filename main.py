import numpy as np
from models import Sequential
from layers import Dense

inputs = np.array([[0, 0, 0],
                   [255, 255, 255],
                   [255, 255, 0],
                   [0, 0, 255],
                   [0, 255, 0]])

inputs_norm = (inputs / 255)

targets = np.array([[1],
                    [0],
                    [0],
                    [1],
                    [0]])

model = Sequential()

model.add(Dense(n_inputs = 3, n_neurons = 3, activation = 'relu'))
model.add(Dense(n_inputs = 3, n_neurons = 1, activation = 'sigmoid'))

model.fit(inputs_norm, targets, eta = 0.01, n_epochs = 100000)