import numpy as np
import matplotlib.pyplot as plt
from numba import njit, vectorize, float64, int64
import pandas as pd
import os

RMAX = 100
BOUND_A, BOUND_B = -np.pi, np.pi
PARENT_DIRECTORY='C:\\Users\\mrzed\\PycharmProjects\\kursovaya\\lorenz'
if not os.path.exists(PARENT_DIRECTORY):
    os.makedirs(PARENT_DIRECTORY)

@njit
def D_L(t, sigma):
    return np.sinh(sigma * np.pi) / (sigma * np.pi * np.cosh(sigma * (t - np.pi)))


@njit
def d_L_m(m, sigma):
    N = RMAX
    result = 0
    coeff = 1 / (2 * N + 1)
    for k in range(1, N + 1):
        result += D_L(2 * np.pi * k * coeff, sigma) * np.cos(2 * np.pi * coeff * m * k)
    return coeff * (D_L(0, sigma) + 2 * result)


@njit
def d_list(sigma):
    coeff_list = []
    for m in range(RMAX + 1):
        coeff_list.append(d_L_m(m, sigma))
    return coeff_list


# @vectorize([(float64, float64)], target="parallel", nopython=True, cache=True)
@np.vectorize
def fi_wave(x, sigma):
    d = d_list(sigma)
    result = 0
    for k in range(-RMAX, RMAX + 1):
        result += d[abs(k)] * sigma * sigma / (sigma * sigma + (x - k) * (x - k))
    return result


def calculate_d(a=0.1, b=2.1, h=0.02):
    sigmas = np.arange(a, b + h, h)
    d = []
    for s in sigmas:
        d.append(d_list(s))

    df = pd.DataFrame(d, index=pd.Index(sigmas, name='sigma'))
    df.to_csv(PARENT_DIRECTORY+'\\d_native.csv')


def calculate_alphas2():
    d = pd.read_csv('gauss_resources/d_native.csv'
                    , header=0
                    , index_col=['sigma']
                    )
    coeffs = pd.read_csv('gauss_resources/sigma_alpha_c_err.csv'
                         , header=0
                         , index_col=['sigma']
                         )
    sigmas = np.array(d.index)
    a = []
    alpha, c = 0, 0
    h_a, h_c = 0.05, 0.05
    for sigma in sigmas:
        _alpha, _c, _ = coeffs.loc[sigma]
        # _alpha, _c, _ = 0., 0., 0.
        alpha, c, err, h_a, h_c = gradient_desc2(sigma, alpha=_alpha, c=_c, h_a=h_a, h_c=h_c)
        # alpha, c, err = coeffs.loc[sigma]
        while np.isnan(alpha) or np.isnan(c):
            print('\n', sigma, '\n')
            ind = np.round((np.round(sigma, 2) - 0.1) / 0.02).astype(int)
            _alpha, _c, _err = coeffs.iloc[ind - 1]
            while np.isnan(_alpha):
                ind -= 1
                _alpha, _c, _err = coeffs.iloc[ind - 1]
            #
            alpha, c, err, h_a, h_c = gradient_desc2(sigma, alpha=_alpha, c=_c, h_a=h_a, h_c=h_c)
            h_a /= 2
            h_c /= 2
        a.append([sigma, alpha, c, err])
    # print(a)
    df = pd.DataFrame(a, columns=['sigma', 'alpha', 'c', 'error'])
    df.set_index('sigma', inplace=True)
    print(df)
    df.to_csv(PARENT_DIRECTORY+'\\sigma_alpha_c_err.csv')
    return df



if __name__ == '__main__':
#     calculate_d()
