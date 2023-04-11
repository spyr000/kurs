import math
from math import exp
import numpy as np
import pandas as pd
from numba import njit

RMAX = 100


def customizable_vectorize(excluded=None):
    if excluded is None:
        excluded = []

    def inner_decorator(func):
        def wrapped(*args, **kwargs):
            return np.vectorize(func, excluded=excluded)(*args, **kwargs)

        return wrapped

    return inner_decorator


def deprecated(func):
    return func


def unused(func):
    return func


def utility(func):
    return func


def moved(func):
    return func


def visual(func):
    return func


def calc(func):
    return func


class Gauss:
    RMAX = 100
    BOUND_A, BOUND_B = -10, 10
    PARENT_DIRECTORY = 'C:\\Users\\mrzed\\PycharmProjects\\kursovaya\\gauss_resources'

    def __init__(self):
        pass

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
            coeff_list.append(Gauss.d_G_k(k, sigma))
        return coeff_list

    @customizable_vectorize(excluded=['d'])
    def phi_wave(x: np.ndarray, sigma: float, d=None):
        fi_val = 0
        if d is None:
            d = pd.read_csv(Gauss.PARENT_DIRECTORY + '//d_native.csv'
                            , header=0
                            , index_col=['sigma']
                            )
            d = np.array(d.loc[sigma])

        for k in range(-RMAX, RMAX + 1):
            fi_val += d[abs(k)] * exp(-((x - k) * (x - k)) / (2 * sigma * sigma))

        return 1 / Gauss.C(sigma) * fi_val


class Lorenz:
    RMAX = 100
    BOUND_A, BOUND_B = -np.pi, np.pi
    PARENT_DIRECTORY = 'C:\\Users\\mrzed\\PycharmProjects\\kursovaya\\lorenz_resources'

    @njit
    def D_L(t: np.ndarray, float, sigma: float):
        return np.sinh(sigma * np.pi) / (sigma * np.pi * np.cosh(sigma * (t - np.pi)))

    @njit
    def d_L_m(m: float, sigma: float):
        N = RMAX
        result = 0
        coeff = 1 / (2 * N + 1)
        for k in range(1, N + 1):
            result += Lorenz.D_L(2 * np.pi * k * coeff, sigma) * np.cos(2 * np.pi * coeff * m * k)
        return coeff * (Lorenz.D_L(0, sigma) + 2 * result)

    @njit
    def d_list(sigma: float):
        coeff_list = []
        for m in range(RMAX + 1):
            coeff_list.append(Lorenz.d_L_m(m, sigma))
        return coeff_list

    # @vectorize([(float64, float64)], target="parallel", nopython=True, cache=True)
    @customizable_vectorize(excluded='d')
    def phi_wave(x: np.ndarray, sigma, d=None):
        if d is None:
            d = pd.read_csv(Lorenz.PARENT_DIRECTORY + '//d_native.csv'
                            , header=0
                            , index_col=['sigma']
                            )
            d = np.array(d.loc[sigma])

        result = 0
        for k in range(-RMAX, RMAX + 1):
            result += d[abs(k)] * sigma * sigma / (sigma * sigma + (x - k) * (x - k))
        return result


@utility
class Container:
    '''очередь из 3 элементов с проверкой равенства
    хранящимся в контейнере элементам вставляемого элемента'''

    def __init__(self):
        self.container = []

    def insert(self, elem):
        if len(self.container) < 3:  # если контейнер заполнен не полностью то вставляем элемент в начало
            self.container.append(elem)
            return False
        elif len(self.container) == 3:  # если контейнер заполнен
            # проверяем равен ли вставляемый элемент какому-то из элементов в контейнере
            if np.isclose(elem, self.container[0]) \
                    or np.isclose(elem, self.container[1]) \
                    or np.isclose(elem, self.container[2]):
                self.container.clear()  # если равен -> очищаем контейнер и возвращаем True
                return True
            else:  # если не равен -> удаляем последний элемент и вставляем новый элемент в начало, возвращаем False
                self.container = [self.container[1], self.container[2], elem]
                return False
        else:
            self.container.clear()
            return False


class Parameters:
    def __init__(self,
                 parent_directory,
                 bounds,
                 d_filename='d_native.csv',
                 coeff_filename='sigma_alpha_c_err.csv',
                 phi_vals_filename='sigma_fi.csv',
                 alpha_derivative=None,
                 c_derivative=None,
                 alpha2_derivative=None,
                 alpha_c_derivative=None,
                 c2_derivative=None,
                 phi_wave=None,
                 d_list_func=None,
                 approx_func=None,
                 file_coeffs=None):
        if file_coeffs is None:
            file_coeffs = ['alpha', 'c']
        self.d_filename = d_filename
        self.parent_directory = parent_directory
        self.c2_derivative = c2_derivative
        self.alpha_c_derivative = alpha_c_derivative
        self.alpha2_derivative = alpha2_derivative
        self.c_derivative = c_derivative
        self.alpha_derivative = alpha_derivative
        self.coeff_filename = coeff_filename
        self.phi_vals_filename = phi_vals_filename
        self.phi_wave = phi_wave
        self.bounds = bounds
        self.d_list_func = d_list_func
        self.approx_func = approx_func
        self.file_coeffs = file_coeffs


k = np.arange(0, RMAX + 1)


# def gauss_approx_func(d_0, x, signchanging=True, k=k):
#     return ((d_0 / 2) * np.exp(x[0] * ((k*x[2]) ** x[1])) + (d_0 / 2) * np.exp(x[3] * ((k*x[5]) ** x[4]))) * (-1.) ** k \
#         if signchanging\
#         else (d_0 / 2) * np.exp(x[0] * ((k*x[2]) ** x[1])) + (d_0 / 2) * np.exp(x[3] * ((k*x[5]) ** x[4]))
# x[2] * np.exp(-(k**2)/(2*(x[3]+x[4])))

# def gauss_approx_func(d_0, x, signchanging=True, k=k):
#     return (d_0 * np.exp(-(k**x[0])/(x[1]))) * (-1.) ** k \
#         if signchanging\
#         else d_0 * np.exp(-(k**x[0])/(x[1]))


# def gauss_approx_func(d_0, x, signchanging=True, k=k):
#     return (x[0] * (k ** 4) + x[1] * (k ** 3) + x[2] * (k ** 2) + x[3] * k + d_0) * (-1.) ** k \
#         if signchanging \
#         else x[0] * (k ** 4) + x[1] * (k ** 3) + x[2] * (k ** 2) + x[3] * k + d_0

def gauss_approx_func(d_0, x, signchanging=True, k=k):
    f = d_0 * np.exp(x[0] * (1 - (1 + (k / x[0]) ** x[1]) ** (1 / x[1])))  # + x[2] * np.sin(np.exp(x[3] * k + x[4]))
    return f * (-1.) ** k \
        if signchanging \
        else f

# def gauss_approx_func(d_0, x, signchanging=True, k=k):
#     f = d_0 * np.exp(
#     if signchanging:
#         return


# def gauss_approx_func(d_0, x, signchanging=True, k=k):
#     f = x[0] * k ** 10 + x[1] * k ** 9 + x[2] * k ** 8 + x[3] * k ** 7 + x[4] * k ** 6 + x[5] * k ** 5 + x[6] * k ** 4 + x[7] * k ** 3 + x[8] * k ** 2 + x[9] * k + d_0
#     f[f < 0] = 0
#     return f * (-1.) ** k \
#         if signchanging \
#         else f


# target_approx_func = lambda x, s=True: x[0] * np.sin(x[1] * k + x[2]) * (-1.) ** k if s else x[0] * np.sin(x[1] * k + x[2])

def target_approx_func(x, s=True):
    f = x[0] * np.sin(x[1] * k + x[2])
    f[f < 0] = 0
    return f * (-1.) ** k if s else f


def lorenz_approx_func(d_0, x, signchanging=True, k=k):
    f = x[0] / (k + (x[0] / d_0) ** (1 / x[1])) ** x[1]
    return f * (-1.) ** k if signchanging else f


@utility
def log(x, base):
    return np.vectorize(math.log)(x, base)


def d_0(sigma):
    result = 0
    for r in range(0, RMAX + 1):
        result += (-1) ** r * np.exp((- (r + 0.5) ** 2) / (2 * sigma * sigma))
    return result


# region derivatives
def alpha_derivative(d, d_pred, d_0, alpha, c, k):
    return 1 / RMAX * (
        np.sum(-d_0 * (k ** c)
               * np.exp(alpha
                        * (k ** c))
               * 2 * (d - d_pred))
    )


def c_derivative(d, d_pred, d_0, alpha, c, k):
    return 1 / RMAX * (
        np.sum(-d_0 * alpha * (k ** c)
               * np.log(k)
               * np.exp(alpha * (k ** c))
               * 2 * (d - d_pred))
    )


def alpha2_derivative(d, d_pred, d_0, alpha, c, k):
    return 1 / RMAX * (
        np.sum(
            -2 * d * (k ** (2 * c)) * d_0 * np.exp((k ** c) * alpha) +
            4 * (k ** (2 * c)) * (d_0 ** 2) * np.exp(2 * (k ** c) * alpha)
        )
    )


def alpha_c_derivative(d, d_pred, d_0, alpha, c, k):
    return 1 / RMAX * \
           np.sum(
               4 * (k ** (2 * c)) * alpha * (d_0 ** 2) * np.exp(
                   2 * (k ** c) * alpha) * np.log(k) \
               - 2 * d * (k ** c) * d_0 * np.log(k) * np.exp((k ** c) * alpha) \
               - 2 * d * (k ** (2 * c)) * alpha * d_0 * np.log(k) * np.exp((k ** c) * alpha) \
               + 2 * (k ** c) * (d_0 ** 2) * np.exp(2 * (k ** c) * alpha) * np.log(k)
           )


def c2_derivative(d, d_pred, d_0, alpha, c, k):
    return 1 / RMAX * \
           np.sum(
               4 * (k ** (2 * c)) * (alpha ** 2) * (d_0 ** 2) * np.exp(2 * (k ** c) * alpha) * (
                       np.log(k) ** 2) \
               - 2 * d * (k ** c) * alpha * d_0 * (np.log(k) ** 2) * np.exp((k ** c) * alpha) \
               - 2 * d * (k ** (2 * c)) * (alpha ** 2) * d_0 * (np.log(k) ** 2) * np.exp(
                   (k ** c) * alpha) \
               + 2 * (k ** c) * alpha * (d_0 ** 2) * np.exp(2 * (k ** c) * alpha) * (
                       np.log(k) ** 2)
           )


# endregion

def get_gauss_setup():
    parameters = Parameters(Gauss.PARENT_DIRECTORY,
                            (Gauss.BOUND_A, Gauss.BOUND_B),
                            d_filename='d_native.csv',
                            coeff_filename='sigma_alpha_c_err.csv',
                            alpha_derivative=alpha_derivative,
                            c_derivative=c_derivative,
                            alpha2_derivative=alpha2_derivative,
                            alpha_c_derivative=alpha_c_derivative,
                            c2_derivative=c2_derivative,
                            phi_wave=Gauss.phi_wave,
                            d_list_func=Gauss.d_list,
                            approx_func=gauss_approx_func,
                            file_coeffs=['a','b'])
    return parameters


def get_lorenz_setup():
    parameters = Parameters(Lorenz.PARENT_DIRECTORY,
                            (Lorenz.BOUND_A, Lorenz.BOUND_B),
                            d_filename='d_native.csv',
                            coeff_filename='sigma_alpha_c_err.csv',
                            alpha_derivative=alpha_derivative,
                            c_derivative=c_derivative,
                            alpha2_derivative=alpha2_derivative,
                            alpha_c_derivative=alpha_c_derivative,
                            c2_derivative=c2_derivative,
                            phi_wave=Lorenz.phi_wave,
                            d_list_func=Lorenz.d_list,
                            approx_func=lorenz_approx_func,
                            file_coeffs=['a', 'b']
                            )
    return parameters
