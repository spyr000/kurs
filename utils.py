import math

import numpy as np

from new_gauss import C

RMAX = 100
BOUND_A, BOUND_B = -10, 10

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
                 d_filename='d_native.csv',
                 coeff_filename='sigma_alpha_c_err.csv',
                 phi_vals_filename='sigma_fi.csv',
                 alpha_derivative=None,
                 c_derivative=None,
                 alpha2_derivative=None,
                 alpha_c_derivative=None,
                 c2_derivative=None,
                 phi_wave=None):
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


@utility
def log(x, base):
    return np.vectorize(math.log)(x, base)


def d_0(sigma):
    result = 0
    for r in range(0, RMAX + 1):
        result += (-1) ** r * np.exp((- (r + 0.5) ** 2) / (2 * sigma * sigma))
    return result


def get_gauss_setup():
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

    def phi_wave(sigma, x, d):
        approx_fi = []

        for xx in x:  # пересчитываем значение приближенной узловой функции
            fi_val = 0
            for k in range(-RMAX, RMAX + 1):
                fi_val += d[abs(k)] * math.exp(-((xx - k) * (xx - k)) / (2 * sigma * sigma))

            fi_val *= 1 / C(sigma)
            approx_fi.append(fi_val)

        return approx_fi

    parameters = Parameters('gauss_resources',
                            d_filename='d_native.csv',
                            coeff_filename='sigma_alpha_c_err.csv',
                            alpha_derivative=alpha_derivative,
                            c_derivative=c_derivative,
                            alpha2_derivative=alpha2_derivative,
                            alpha_c_derivative=alpha_c_derivative,
                            c2_derivative=c2_derivative,
                            phi_wave=phi_wave)
    return parameters
