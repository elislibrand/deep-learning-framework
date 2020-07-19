import numpy as np
from framework.models import Sequential
from framework.layers import Dense
from framework.optimizers import SGD
from framework.regularizers import L1, L2, MaxNorm

def get_data():
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
    
    return inputs_norm, targets

def main():
    inputs, targets = get_data()
    
    # 3-3-1 (classifier)
    model = Sequential([
        Dense(3, activation = 'relu', regularizers = [L2(amount = 1e-4)]),
        Dense(1, activation = 'sigmoid', regularizers = [L2(amount = 1e-4)])
    ])
    
    model.compile(optimizer = SGD(learning_rate = 1e-3))
    
    model.fit(inputs, targets, batch_size = 1, n_epochs = 10, seed = 0)
    
if __name__ == "__main__":
    #if wmi.WMI().Win32_VideoController()[0].AdapterCompatibility.lower() == 'nvidia':
    #    import cupy as np
    #else:
    #    import numpy as np

    main()