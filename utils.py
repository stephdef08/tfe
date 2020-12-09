import torch
import numpy as np

#Replace with something more efficient
def binarize(tensor):
    binary_rep = np.zeros((tensor.shape[0], tensor.shape[1]))
    row = binary_rep.shape[0]
    col = binary_rep.shape[1]
    for y in range(row):
        for i in range(col):
            binary_rep[y][i] = 0 if tensor[y][i] < .5 else 1

    return binary_rep
