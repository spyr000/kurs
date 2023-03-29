import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utils import RMAX, Parameters, gauss_approx_func

mse = lambda x, y: 1 / len(x) * np.sum((x - y) ** 2)


class GaussLoss:
    def __init__(self, d, k, approx_func):
        self.d = d
        # self.approx_func = lambda x: d[0] * np.exp(x[0] * (k ** x[1]))
        self.approx_func = lambda x: approx_func(d[0], x, False)
        # self.approx_func = lambda x: d[0] * np.exp(x[0] * (k ** 2))
        self.mse = lambda x: mse(d, self.approx_func(x))


class LorenzLoss:
    def __init__(self, d, k):
        self.d = d
        self.approx_func = lambda x: x[0] / (k ** x[1])


def calculate_new_vals(parameters: Parameters):
    dff = pd.read_csv(parameters.parent_directory + '\\' + parameters.d_filename
                      , header=0
                      , index_col=['sigma']
                      )
    sigmas = np.array(dff.index)
    vals = []
    for sigma in sigmas:
        k = np.arange(0, RMAX + 1)
        d = np.abs(np.array(dff.loc[sigma]))
        loss_func = GaussLoss(d, k, parameters.approx_func)
        result = minimize(loss_func.mse, x0=np.array([0, 0, 0.001, 0, 0, 0.001]), method='nelder-mead')
        # result = minimize(loss_func.mse, x0=np.array([0]), method='nelder-mead')
        print(result)
        coeffs = result.x.tolist()

        vals.append([sigma] + coeffs + [mse(loss_func.d, loss_func.approx_func(coeffs))])
        # vals.append([sigma] + coeffs + [2] + [mse(loss_func.d, loss_func.approx_func(coeffs))])

    df = pd.DataFrame(vals, columns=['sigma'] + parameters.file_coeffs + ['error'])
    # df = pd.DataFrame(vals, columns=['sigma', 'a', 'b', 'c', 'error'])
    df.set_index('sigma', inplace=True)
    print(df)
    df.to_csv(parameters.parent_directory + '\\' + parameters.coeff_filename)
