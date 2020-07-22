import numpy as np

inputs = np.array([[255, 138, 128],
                   [255, 188, 175],
                   [200, 90, 84],
                   [255, 82, 82],
                   [255, 134, 127],
                   [197, 14, 41],
                   [255, 23, 68],
                   [255, 97, 111],
                   [196, 0, 29],
                   [213, 0, 0],
                   [255, 81, 49],
                   [155, 0, 0],
                   [255, 128, 171],
                   [255, 178, 221],
                   [201, 79, 124],
                   [255, 64, 129],
                   [255, 121, 176],
                   [198, 0, 85],
                   [245, 0, 87],
                   [255, 89, 131],
                   [187, 0, 47],
                   [197, 17, 98],
                   [253, 85, 143],
                   [142, 0, 56],
                   [234, 128, 252],
                   [255, 178, 255],
                   [182, 79, 200],
                   [224, 64, 251],
                   [255, 121, 255],
                   [170, 0, 199],
                   [213, 0, 249],
                   [255, 91, 255],
                   [158, 0, 197],
                   [170, 0, 255],
                   [226, 84, 255],
                   [114, 0, 202],
                   [179, 136, 255],
                   [231, 185, 255],
                   [128, 90, 203],
                   [124, 77, 255],
                   [180, 124, 255],
                   [63, 29, 203],
                   [101, 31, 255],
                   [162, 85, 255],
                   [1, 0, 202],
                   [98, 0, 234],
                   [157, 70, 255],
                   [10, 0, 182],
                   [140, 158, 255],
                   [192, 207, 255],
                   [88, 112, 203],
                   [83, 109, 254],
                   [143, 155, 255],
                   [0, 67, 202],
                   [61, 90, 254],
                   [129, 135, 255],
                   [0, 49, 202],
                   [48, 79, 254],
                   [122, 124, 255],
                   [0, 38, 202],
                   [130, 177, 255],
                   [182, 227, 255],
                   [77, 130, 203],
                   [68, 138, 255],
                   [131, 185, 255],
                   [0, 94, 203],
                   [41, 121, 255],
                   [117, 167, 255],
                   [0, 78, 203],
                   [41, 98, 255],
                   [118, 143, 255],
                   [0, 57, 203],
                   [128, 216, 255],
                   [181, 255, 255],
                   [73, 167, 204],
                   [64, 196, 255],
                   [130, 247, 255],
                   [0, 148, 204],
                   [0, 176, 255],
                   [105, 226, 255],
                   [0, 129, 203],
                   [0, 145, 234],
                   [100, 193, 255],
                   [0, 100, 183],
                   [132, 255, 255],
                   [186, 255, 255],
                   [75, 203, 204],
                   [24, 255, 255],
                   [118, 255, 255],
                   [0, 203, 204],
                   [0, 229, 255],
                   [110, 255, 255],
                   [0, 178, 204],
                   [0, 184, 212],
                   [98, 235, 255],
                   [0, 136, 163]])

targets = np.array([[0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [1],
                    [0],
                    [0],
                    [1],
                    [1],
                    [0],
                    [1],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [1],
                    [0],
                    [0],
                    [1],
                    [1],
                    [0],
                    [1],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [1],
                    [0],
                    [0],
                    [1],
                    [1],
                    [0],
                    [1],
                    [0],
                    [0],
                    [1],
                    [1],
                    [0],
                    [1],
                    [1],
                    [0],
                    [1],
                    [1],
                    [0],
                    [1],
                    [0],
                    [0],
                    [1],
                    [0],
                    [0],
                    [1],
                    [1],
                    [0],
                    [1],
                    [1],
                    [0],
                    [1],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [1],
                    [0],
                    [0],
                    [1],
                    [1],
                    [0],
                    [1],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [1],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0]])

def get():
    return inputs, targets

def get_normalized():
    return (inputs / 255), targets