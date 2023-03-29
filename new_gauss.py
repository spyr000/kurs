from math import exp
import numpy as np
import pandas as pd
from numba import njit

RMAX = 100
BOUND_A, BOUND_B = -10, 10
PARENT_DIRECTORY = 'C:\\Users\\mrzed\\PycharmProjects\\kursovaya\\gauss_resources'


@njit(cache=True)
def C(sigma: float) -> float:
    result = 0
    for r in range(-RMAX, RMAX + 1):
        result += (4 * r + 1) * np.exp(-(2 * r + 0.5) * (2 * r + 0.5) / (2 * sigma * sigma))
    return result


@njit(cache=True)
def d_G_k(k: int, sigma: float) -> float:
    result = 0
    for r in range(abs(k), abs(k) + RMAX + 1):
        result += (-1) ** r * np.exp((k * k - (r + 0.5) ** 2) / (2 * sigma * sigma))
    return result


@njit(cache=True)
def d_list(sigma: float) -> list:
    coeff_list = []
    for k in range(RMAX + 1):
        coeff_list.append(d_G_k(k, sigma))
    return coeff_list


# @customizable_vectorize(excluded=['sigma'])
@np.vectorize
def phi_wave(x: np.ndarray, sigma: float, d=None):
    fi_val = 0
    if d is None:
        d = pd.read_csv(PARENT_DIRECTORY + '//d_native.csv'
                        , header=0
                        , index_col=['sigma']
                        )
        d = np.array(d.loc[sigma])

    for k in range(-RMAX, RMAX + 1):
        fi_val += d[abs(k)] * exp(-((x - k) * (x - k)) / (2 * sigma * sigma))

    return 1 / C(sigma) * fi_val
