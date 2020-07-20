import numpy as np
from framework.datasets import rgb
from framework.models import Sequential
from framework.layers import Dense
from framework.optimizers import SGD
from framework.initializers import GlorotNormal, HeUniform
from framework.regularizers import L1, L2

def get_data():
    return rgb.get_normalized()

def main():
    inputs, targets = get_data()
    
    # 3-3-3-1 (classifier)
    model = Sequential([
        Dense(3, activation = 'relu', initializer = HeUniform(), regularizers = [L1(amount = 1e-4)]),
        Dense(3, activation = 'relu', initializer = HeUniform(), regularizers = [L1(amount = 1e-4)]),
        Dense(1, activation = 'sigmoid', initializer = HeUniform(), regularizers = [L1(amount = 1e-4)])
    ])
    
    model.compile(optimizer = SGD(learning_rate = 0.01, momentum = 0.9))
    
    model.fit(inputs, targets, batch_size = 4, n_epochs = 100)
    
if __name__ == "__main__":
    #if wmi.WMI().Win32_VideoController()[0].AdapterCompatibility.lower() == 'nvidia':
    #    import cupy as np
    #else:
    #    import numpy as np

    main()