import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def inv_sigmoid(w):
    return np.log(w/(1-w))
    