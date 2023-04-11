import numpy as np

mse = lambda x, y: 1 / len(x) * np.sum((x - y) ** 2)

RMAX = 100
k = np.arange(0, RMAX + 1)

def get_mse(true_values, pred_func):
    return lambda x: mse(true_values, pred_func(x))

def signchanging(func):
    return func * (-1.) ** k
