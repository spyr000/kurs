import numpy as np
from numba import njit, vectorize, float64, int64
import pandas as pd
import os


RMAX = 100
BOUND_A, BOUND_B = -np.pi, np.pi
PARENT_DIRECTORY='C:\\Users\\mrzed\\PycharmProjects\\kursovaya\\lorenz_resources'
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
def phi_wave(x, sigma, d=None):
    if d is None:
        d = pd.read_csv(PARENT_DIRECTORY + '//d_native.csv'
                        , header=0
                        , index_col=['sigma']
                        )
        d = np.array(d.loc[sigma])

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


if __name__ == '__main__':
    p = utils.get_lorenz_setup()
#     calculate_d()
