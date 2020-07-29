import numpy as np
from framework.datasets import rgb
from framework.models import Sequential
from framework.layers import Dense
from framework.optimizers import RMSProp, SGD, Adam
from framework.activations import ReLU, Sigmoid
from framework.losses import BCE
from framework.initializers import HeUniform
from framework.regularizers import L1, L2
from framework.constraints import MaxNorm

def get_data():
    inputs, targets = rgb.get_normalized()
    
    print('Class [0]: {}\nClass [1]: {}\n'.format(np.around(np.sum(targets == 0) / len(targets), 2),
                                                  np.around(np.sum(targets == 1) / len(targets), 2)))

    return inputs, targets

def get_color():
    user_input = input('\nInput:  ')
    
    if ',' in user_input:
        color = np.asarray([int(x) for x in user_input.split(',')])
    else:
        color = np.asarray([int(user_input.lstrip('#')[x:x + 2], 16) for x in (0, 2, 4)])
    
    return color / 255

def main():
    inputs, targets = get_data()
    
    # 3-3-3-1 (classifier)
    model = Sequential([
        Dense(3, activation = ReLU(), initializer = HeUniform(), regularizers = [L1(amount = 1e-4)], constraints = [MaxNorm()]),
        Dense(3, activation = ReLU(), initializer = HeUniform(), regularizers = [L1(amount = 1e-4)], constraints = [MaxNorm()]),
        Dense(1, activation = Sigmoid(), initializer = HeUniform(), regularizers = [L1(amount = 1e-4)], constraints = [MaxNorm()])
    ])
    
    model.compile(optimizer = Adam(learning_rate = 1e-3), loss = BCE())
    
    model.fit(inputs, targets, batch_size = 32, n_epochs = 500)
    
    # Prediction
    prediction = int(np.squeeze(model.predict(get_color())))
    
    print('Output: {} ({})'.format(prediction, 'black' if prediction == 0 else 'white'))

if __name__ == "__main__":
    #if wmi.WMI().Win32_VideoController()[0].AdapterCompatibility.lower() == 'nvidia':
    #    import cupy as np
    #else:
    #    import numpy as np

    main()