import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

from MseConstruct import get_mse
from utils import RMAX, Parameters, gauss_approx_func, target_approx_func

mse = lambda x, y: 1 / len(x) * np.sum((x - y) ** 2)


# class GaussLoss:
#     def __init__(self, d, k, approx_func):
#         self.d = d
#         # self.approx_func = lambda x: d[0] * np.exp(x[0] * (k ** x[1]))
#         self.approx_func = lambda x: approx_func(d[0], x, False)
#         # self.approx_func = lambda x: d[0] * np.exp(x[0] * (k ** 2))
#         self.mse = lambda x: mse(d, self.approx_func(x))


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
        # loss_func = GaussLoss(d, k, parameters.approx_func)
        # result = minimize(loss_func.mse, x0=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), method='Nelder-Mead', tol=10e-16)
        true_vals = np.log(d / d[0]) + (k / (2 * sigma * sigma))
        count_func = lambda x: np.max(true_vals) * (1 - np.exp(-x[0] * k))

        result = minimize(get_mse(true_vals, count_func, d[0]),
                          x0=np.array([0]), method="powell", tol=10e-16)
        print(result)
        coeffs = result.x.tolist()

        vals.append([sigma] + coeffs + [mse(d, parameters.approx_func(d[0], coeffs, False))])
        # vals.append([sigma] + coeffs + [2] + [mse(loss_func.d, loss_func.approx_func(coeffs))])

    df = pd.DataFrame(vals, columns=['sigma'] + parameters.file_coeffs + ['error'])
    # df = pd.DataFrame(vals, columns=['sigma', 'a', 'b', 'c', 'error'])
    df.set_index('sigma', inplace=True)
    print(df)
    df.to_csv(parameters.parent_directory + '\\' + parameters.coeff_filename)


def calculate_new_vals_two_steps(parameters: Parameters):
    dff = pd.read_csv(parameters.parent_directory + '\\' + parameters.d_filename
                      , header=0
                      , index_col=['sigma']
                      )
    sigmas = np.array(dff.index)
    vals = []
    k = np.arange(0, RMAX + 1)

    for sigma in sigmas:
        d = np.abs(np.array(dff.loc[sigma]))
        loss_func = GaussLoss(d, k, parameters.approx_func)
        result = minimize(loss_func.mse, x0=np.array([0, 0]), method='bfgs')
        coeffs = result.x.tolist()
        target = d - gauss_approx_func(d[0], coeffs, signchanging=False)

        result = minimize(lambda x: mse(target, target_approx_func(x)), x0=np.array([0, 0, 0]), method='nelder-mead',
                          tol=1e-11)
        coeffs2 = coeffs + result.x.tolist()
        vals.append([sigma] + coeffs2 + [
            mse(d - target, loss_func.approx_func(coeffs) + target_approx_func(result.x.tolist()))])

    df = pd.DataFrame(vals, columns=['sigma'] + parameters.file_coeffs + ['error'])
    df.set_index('sigma', inplace=True)
    print(df)
    df.to_csv(parameters.parent_directory + '\\' + parameters.coeff_filename)
