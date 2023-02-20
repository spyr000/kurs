import math
from functools import partial
from math import exp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm as tqdm
from matplotlib import animation
from matplotlib.widgets import Slider
from numba import njit
from tqdm import tqdm
from utils import deprecated, unused, utility

RMAX = 100
BOUND_A, BOUND_B = -10, 10
PARENT_DIRECTORY='C:\\Users\\mrzed\\PycharmProjects\\kursovaya\\gauss'

decorator = partial(np.vectorize, excluded=['sigma'])

@unused
@utility
def customizable_vectorize(excluded=None):
    if excluded is None:
        excluded = []

    def inner_decorator(func):
        def wrapped(*args, **kwargs):
            return np.vectorize(func(*args, **kwargs), excluded=excluded)

        return wrapped
    return inner_decorator


@utility
def log(x, base):
    return np.vectorize(math.log)(x, base)


@unused
@utility
def approx_func(sigma, alpha, c, k=np.arange(0, RMAX + 1)):
    return d_G_k(0, sigma) \
           * np.exp(alpha * (k ** c)) \
           * (-1.) \
           ** k


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
def fi_wave(x: np.ndarray, sigma: float):
    fi_val = 0
    # d = d_list(sigma)
    d = pd.read_csv('gauss/d_native.csv'
                    , header=0
                    , index_col=['sigma']
                    )
    d = np.array(d.loc[sigma])

    for k in range(-RMAX, RMAX + 1):
        fi_val += d[abs(k)] * exp(-((x - k) * (x - k)) / (2 * sigma * sigma))

    return 1 / C(sigma) * fi_val



@np.vectorize
def approx_fi_wave2(x: np.ndarray, sigma: float):
    kk = np.arange(0, RMAX + 1)
    fi_val = 0
    # d = d_list(sigma)
    # print(d)
    d_0 = pd.read_csv('gauss/d_native.csv'
                      , header=0
                      , index_col=['sigma']
                      ).loc[sigma].iat[0]
    dff = pd.read_csv('gauss/sigma_alpha_c_err.csv'
                      , header=0
                      , index_col=['sigma']
                      )

    alpha, c, error = dff.loc[sigma]
    if alpha is None or c is None or error is None:
        print('DDD')
        ind = np.round((np.round(sigma, 2) - 0.1) / 0.02).astype(int)
        alpha, c, error = dff.iloc[ind - 1]

    d = d_0 \
        * np.exp(alpha *
                 (kk ** c)) \
        * (-1.) ** kk

    for k in range(-RMAX, RMAX + 1):
        # print(d[abs(k)])
        fi_val += d[abs(k)] * exp(-((x - k) * (x - k)) / (2 * sigma * sigma))

    return 1 / C(sigma) * fi_val


@unused
@deprecated
def gradient_desc(sigma, alpha=0, lr=1e-4):
    d = pd.read_csv('gauss/d_native.csv'
                    , header=0
                    , index_col=['sigma']
                    )
    d = np.abs(np.array(d.loc[sigma]))
    k = np.arange(0, RMAX + 1)
    d_0 = d[0]
    mse = lambda x, y: 1 / len(x) * np.sum((x - y) ** 2)
    eps = 1
    prev_eps = 1
    cnt = 0
    max_repeat = 10
    max_iter = 10_000
    h = 0.5
    flag = True
    print('сигма:', sigma)
    i = 0
    while eps > lr and flag and i < max_iter:
        d_pred = d_0 * np.exp(alpha * np.abs(k))
        eps = mse(d, d_pred)
        if eps == prev_eps:
            # alpha -= 0.0002
            cnt += 1
        if cnt > max_repeat:
            cnt = 0
            flag = False
        else:
            grad = 1 / RMAX * (np.sum(-2 * d_0 * np.exp(alpha * np.abs(k)) * (d - d_pred)))
            alpha -= h * grad
            print('\tальфа:', alpha, 'ошибка:', eps)
            if np.abs(prev_eps - eps) < 1e-14:
                flag = False
            prev_eps = eps
            i += 1
    return alpha, eps


@utility
def is_pos_def(x):
    '''проверка на положительную определенность матрицы 2 x 2'''
    return x[0, 0] > 0 and (x[0, 0] * x[1, 1] - x[0, 1] * x[1, 0]) > 0


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


def gradient_desc2(sigma, alpha=0, c=0, lr=1e-8, h_a=0.005, h_c=0.005):
    container = Container()

    d = pd.read_csv('gauss/d_native.csv'
                    , header=0
                    , index_col=['sigma']
                    )
    # cчитываем значения | d(sigma) | из файла
    d = np.abs(np.array(d.loc[sigma]))[:np.round(sigma * 10).astype(int) + 1]
    d_0 = d[0]  # записываем в переменную значение d_0
    d = d[1:]  # в качестве целевых значений будем использовать остальные коэффициенты из набора | d(sigma) |
    k = np.arange(1, RMAX + 1)[
        :np.round(sigma * 10).astype(int)]  # номера коэффициентов (обнуляем коэффиенты d_i, i > [sigma*10])

    mse = lambda x, y: 1 / len(x) * np.sum((x - y) ** 2)  # функция ошибки
    eps = 1  # начальное значение ошибки
    cnt = 0  # счетчик повторений значений ошибок
    max_repeat = 10  # число максимального количества повторений значений ошибок
    max_iter = 100_000  # число максимального количества итераций поиска новых значений коэффициентов alpha и c

    flag = True  # флаг выхода из цикла
    print('сигма:', sigma)
    i = 0
    while eps > lr and flag:

        d_pred = d_0 * np.exp(alpha * (k ** c))  # считаем начальное значение приближенных коэффициентов d_
        eps = mse(d, d_pred)  # считаем ошибку вычислений
        if container.insert(eps):  # если ошибка равна одной из ошибок на предыдущих итерациях
            cnt += 1  # увеличиваем значение счетчика
        if cnt > max_repeat:  # если значение счетчика превышает критическое
            cnt = 0
            flag = False  # выходим из цикла вычислений
        else:  # иначе
            try:
                # Находим минимум функции ошибки MSE:
                # считаем частную производную ф-ции MSE по переменной alpha
                alpha_derivative = 1 / RMAX * (
                    np.sum(-d_0 * (k ** c)
                           * np.exp(alpha
                                    * (k ** c))
                           * 2 * (d - d_pred))
                )
                # считаем частную производную ф-ции MSE по переменной c
                c_derivative = 1 / RMAX * (
                    np.sum(-d_0 * alpha * (k ** c)
                           * np.log(k)
                           * np.exp(alpha * (k ** c))
                           * 2 * (d - d_pred)))
                # считаем 2-ю частную производную ф-ции MSE по переменной alpha
                alpha2_derivative = 1 / RMAX * \
                                    np.sum(
                                        -2 * d * (k ** (2 * c)) * d_0 * np.exp((k ** c) * alpha) \
                                        + 4 * (k ** (2 * c)) * (d_0 ** 2) * np.exp(2 * (k ** c) * alpha)
                                    )
                # считаем 2-ю частную производную ф-ции MSE по переменным alpha и c
                alpha_c_derivative = 1 / RMAX * \
                                     np.sum(
                                         4 * (k ** (2 * c)) * alpha * (d_0 ** 2) * np.exp(
                                             2 * (k ** c) * alpha) * np.log(k) \
                                         - 2 * d * (k ** c) * d_0 * np.log(k) * np.exp((k ** c) * alpha) \
                                         - 2 * d * (k ** (2 * c)) * alpha * d_0 * np.log(k) * np.exp((k ** c) * alpha) \
                                         + 2 * (k ** c) * (d_0 ** 2) * np.exp(2 * (k ** c) * alpha) * np.log(k)
                                     )
                # считаем 2-ю частную производную ф-ции MSE по переменной с
                c2_derivative = 1 / RMAX * \
                                np.sum(
                                    4 * (k ** (2 * c)) * (alpha ** 2) * (d_0 ** 2) * np.exp(2 * (k ** c) * alpha) * (
                                            np.log(k) ** 2) \
                                    - 2 * d * (k ** c) * alpha * d_0 * (np.log(k) ** 2) * np.exp((k ** c) * alpha) \
                                    - 2 * d * (k ** (2 * c)) * (alpha ** 2) * d_0 * (np.log(k) ** 2) * np.exp(
                                        (k ** c) * alpha) \
                                    + 2 * (k ** c) * alpha * (d_0 ** 2) * np.exp(2 * (k ** c) * alpha) * (
                                            np.log(k) ** 2)
                                )
                # составляем матрицу Гессе
                hessian = np.array([
                    [alpha2_derivative, alpha_c_derivative],
                    [alpha_c_derivative, c2_derivative]
                ])
                if is_pos_def(hessian):  # если матрица Гессе положительно определена
                    _hessian = np.linalg.inv(hessian)  # находим обратную матрицу для гессиана
                    grad = np.array([[alpha_derivative], [c_derivative]])  # составляем вектор градиента
                    _d = (_hessian @ grad).flatten()  # считаем значения прирощений коэффициентов alpha и c
                    alpha -= h_a * _d[0]
                    c -= h_c * _d[1]
                else:  # если матрица Гессе не положительно определена
                    # считаем новые значения alpha и с учитывая только первую производную ф-ции MSE
                    alpha -= h_a * alpha_derivative
                    c -= h_c * c_derivative
                print(f'\t{i}) альфа:', alpha, 'степень:', c, 'ошибка:', eps, 'сигма:', sigma)
                if i > max_iter:  # если количество итераций превысило критическое значение
                    flag = False  # выходим из цикла
                i += 1  # увеличиваем счетчик итераций
            except OverflowError:  # если вычисленное значение приближенных коэффициентов d является неадекватным
                print("OVERFLOW!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                break  # выходим из цикла

    return alpha, c, eps, h_a, h_c


@deprecated
def gradient_desc3(sigma, alpha=0, c=0, lr=1e-6, h_a=0.5, h_c=0.5):
    d = pd.read_csv('gauss/d_native.csv'
                    , header=0
                    , index_col=['sigma']
                    )
    d = np.abs(np.array(d.loc[sigma]))[1:]
    k = np.arange(1, RMAX + 1)
    d_0 = d[0]
    mse = lambda x, y: 1 / len(x) * np.sum((x - y) ** 2)
    eps = 1
    prev_eps = 1
    cnt = 0
    max_repeat = 10
    max_iter = 100_000

    flag = True
    print('сигма:', sigma)
    i = 0
    while eps > lr and flag:

        d_pred = d_0 * np.exp(alpha * (k ** c))
        eps = mse(d, d_pred)
        if eps == prev_eps:
            cnt += 1
        if cnt > max_repeat:
            cnt = 0
            flag = False
        else:
            alpha_derivative = 1 / RMAX * (
                np.sum(-d_0 * (k ** c)
                       * np.exp(alpha
                                * (k ** c))
                       * 2 * (d - d_pred))
            )
            c_derivative = 1 / RMAX * (
                np.sum(-d_0 * alpha * (k ** c)
                       * np.log(k)
                       * np.exp(alpha * (k ** c))
                       * 2 * (d - d_pred)))

            alpha -= h_a * alpha_derivative
            c -= h_c * c_derivative

            # if c > 4:
            #     c = 4
            print(f'\t{i}) альфа:', alpha, 'степень:', c, 'ошибка:', eps, 'сигма:', sigma)
            if i > max_iter:
                flag = False
            prev_eps = eps
            i += 1

    return alpha, c, eps, h_a, h_c

@unused
@deprecated
def newton_gauss(sigma, h_a, h_c, alpha=0, c=0, lr=1e-6):
    d = pd.read_csv('gauss/d_native.csv'
                    , header=0
                    , index_col=['sigma']
                    )
    d = np.abs(np.array(d.loc[sigma]))[1:]
    k = np.arange(1, RMAX + 1)
    d_0 = d[0]
    mse = lambda x, y: 1 / len(x) * np.sum((x - y) ** 2)
    eps = 1
    prev_eps = 1
    cnt = 0
    max_repeat = 10
    max_iter = 40_000

    flag = True
    print('сигма:', sigma)
    i = 0
    while eps > lr and flag:
        d_pred = d_0 * np.exp(alpha * (k ** c))

        eps = mse(d, d_pred)
        if eps == prev_eps:
            cnt += 1
        if cnt > max_repeat:
            cnt = 0
            flag = False
        else:
            alpha_derivative = -d_0 * (k ** c) \
                               * np.exp(alpha * (k ** c)) \
                               * 2 * (d - d_pred)

            c_derivative = -d_0 * alpha * (k ** c) \
                           * np.log(k) \
                           * np.exp(alpha * (k ** c)) \
                           * 2 * (d - d_pred)

            jacobian = np.array([
                alpha_derivative.tolist(),
                c_derivative.tolist()
            ]).T

            while True:
                try:
                    _d = np.linalg.pinv(jacobian) @ \
                         np.hstack((r := (d_pred - d).reshape(-1, 1), r))
                    break
                except np.linalg.LinAlgError:
                    continue
            # print(_d)
            alpha -= h_a * _d[0, 0]
            c -= h_c * _d[1, 1]

            # alpha_derivative -= h_a * alpha2_derivative + 0.5 * (h_a ** 2) *
            # c2_derivative -= h_c * c2_derivative
            # alpha -= h_a * alpha_derivative
            # if alpha2_derivative > 0:
            #     alpha -= 0.5 * (h_a ** 2) * alpha2_derivative
            #
            # c -= h_c * c_derivative
            # if c2_derivative > 0:
            #     c2_derivative -= 0.5 * (h_c ** 2) * c2_derivative
            if c > 4:
                c = 4
            if alpha < -20:
                alpha = -20
            if alpha > 0:
                alpha = 0
            print(f'\t{i}) альфа:', alpha, 'степень:', c, 'ошибка:', eps, 'сигма:', sigma)
            if i > max_iter:
                flag = False
            prev_eps = eps
            i += 1

    return alpha, c, eps, h_a, h_c


@deprecated
def calculate_alphas():
    d = pd.read_csv('gauss/d_native.csv'
                    , header=0
                    , index_col=['sigma']
                    )
    sigmas = np.array(d.index)
    a = []
    for sigma in sigmas:
        alpha, err = gradient_desc2(sigma)
        a.append([sigma, alpha, err])
    df = pd.DataFrame(a, columns=['sigma', 'alpha', 'error'])
    df.set_index('sigma', inplace=True)
    print(df)
    df.to_csv('gauss/sigma_alpha_err.csv')
    return df


def calculate_alphas2():
    d = pd.read_csv('gauss/d_native.csv'
                    , header=0
                    , index_col=['sigma']
                    )
    coeffs = pd.read_csv('gauss/sigma_alpha_c_err.csv'
                         , header=0
                         , index_col=['sigma']
                         )
    sigmas = np.array(d.index)
    a = []
    alpha, c = 0, 0
    h_a, h_c = 0.05, 0.05
    for sigma in sigmas:
        _alpha, _c, _ = coeffs.loc[sigma]  # начальные значения alpha и c
        # _alpha, _c, _ = 0., 0., 0.
        alpha, c, err, h_a, h_c = gradient_desc2(sigma, alpha=_alpha, c=_c, h_a=h_a,
                                                 h_c=h_c)  # получаем новые значения коэффициентов
        # alpha, c, err = coeffs.loc[sigma]
        while np.isnan(alpha) or np.isnan(c):  # если коэффициенты равны NaN
            print('\n', sigma, '\n')
            ind = np.round((np.round(sigma, 2) - 0.1) / 0.02).astype(int)  # расчитываем индекс сигмы из таблицы
            _alpha, _c, _err = coeffs.iloc[ind - 1]  # получаем значения из предыдущей строки с коэффициентами
            while np.isnan(_alpha):  # если эти значения тоже NaN
                ind -= 1
                _alpha, _c, _err = coeffs.iloc[ind - 1]  # получаем значения из предыдущей строки с коэффициентами
            alpha, c, err, h_a, h_c = gradient_desc2(sigma, alpha=_alpha, c=_c, h_a=h_a,
                                                     h_c=h_c)  # пересчитыеваем значения коэффициентов начиная с другого начального приближения
            # половиним шаги
            h_a /= 2
            h_c /= 2
        # записываем коэффициенты в список
        a.append([sigma, alpha, c, err])
    df = pd.DataFrame(a, columns=['sigma', 'alpha', 'c', 'error'])
    df.set_index('sigma', inplace=True)
    print(df)
    df.to_csv('gauss/sigma_alpha_c_err.csv')
    return df


@deprecated
@np.vectorize
def approx_fi_wave(x: np.ndarray, sigma: float):
    '''функция расчета приближенного значения коэфициентов d'''
    kk = np.arange(0, RMAX + 1)  # список номеров коэффициентов
    fi_val = 0
    d_0 = pd.read_csv('gauss/d_native.csv'
                      , header=0
                      , index_col=['sigma']
                      ).loc[sigma].iat[0]
    d = pd.read_csv('gauss/sigma_alpha_err.csv'
                    , header=0
                    , index_col=['sigma']
                    )

    alpha = np.array(d.at[sigma, 'alpha'])
    d = d_0 * np.exp(alpha * np.abs(kk)) * (-1.) ** kk

    for k in range(-RMAX, RMAX + 1):
        fi_val += d[abs(k)] * exp(-((x - k) * (x - k)) / (2 * sigma * sigma))

    return 1 / C(sigma) * fi_val


def calculate_fi():
    df = pd.read_csv('gauss/sigma_alpha_c_err.csv'
                     , header=0
                     , index_col=['sigma']
                     )

    fi_list, approx_fi_list = [], []
    x = np.linspace(BOUND_A, BOUND_B, 1000)

    for sigma in tqdm(df.index):
        y = fi_wave(x, sigma)
        y1 = approx_fi_wave2(x, sigma)
        # fi_list.append(y.tolist())
        approx_fi_list.append(y1.tolist())

    # df1 = pd.DataFrame(fi_list, index=df.index)
    # df1.to_csv('gauss/sigma_fi.csv')
    df1 = pd.DataFrame(approx_fi_list, index=df.index)
    df1.to_csv('gauss/sigma_approx_fi2.csv')


def calculate_d(a=0.1, b=2.1, h=0.02):
    df_list = []
    sigmas = np.arange(a, b + h, h)
    for sigma in sigmas:
        df_list.append(d_list(sigma))

    df = pd.DataFrame(df_list, index=pd.Index(sigmas, name='sigma'))
    df.to_csv('gauss/d_native.csv')


@deprecated
def play_animation():
    df = pd.read_csv('gauss/sigma_alpha_err.csv'
                     , header=0
                     , index_col=['sigma']
                     )
    df1 = pd.read_csv('gauss/d_native.csv'
                      , header=0
                      , index_col=['sigma']
                      )

    k = np.arange(0, RMAX + 1)
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 101), ylim=(0, 1))
    line1, = ax.plot([], [], lw=2)
    line2, = ax.plot([], [], lw=2)
    text2 = ax.text(0.55, 0.85, '', transform=ax.transAxes)

    d_list = []
    d_pred_list = []

    def animate(i):
        x = d_list[i]
        y = d_pred_list[i]

        line1.set_data(k, x)
        line1.set_marker('.')
        line1.set_label('d')
        text2.set_text(('ошибка: ' + str(df.iat[i, 1])))
        line2.set_data(k, y, )
        line2.set_marker('.')
        plt.legend()
        return line1, line2, text2,

    for i in df.index:
        alpha = df.at[i, 'alpha']
        d = np.abs(np.array(df1.loc[i]))
        d_pred = d[0] * np.exp(alpha * np.abs(k))
        d_list.append(d)
        d_pred_list.append(d_pred)
        # plt.plot(k, d, label='d')
        # plt.plot(k, d_pred, label=('ошибка: ' + str(df.at[i, 'error'])))
        # plt.legend()
        # plt.grid()
        # plt.show()

    anim = animation.FuncAnimation(fig, animate,
                                   frames=101,
                                   interval=200,
                                   blit=True)
    plt.grid()
    plt.show()
    # anim.save('gif.gif',writer='imagemagick')


def sigma_d_slider():
    df = pd.read_csv('gauss/sigma_alpha_c_err.csv'
                     , header=0
                     , index_col=['sigma']
                     )
    df1 = pd.read_csv('gauss/d_native.csv'
                      , header=0
                      , index_col=['sigma']
                      )
    sig = np.array(df.index)
    k = np.arange(0, RMAX + 1)
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 101), ylim=(0, 1))
    line1, = ax.plot([], [], lw=2)
    line2, = ax.plot([], [], lw=0.5)
    text2 = ax.text(0.55, 0.85, '', transform=ax.transAxes)
    text3 = ax.text(0.55, 0.75, '', transform=ax.transAxes)

    d_list = []
    d_pred_list = []

    axsigm = plt.axes([0.15, 0.01, 0.65, 0.03])
    axalpha = plt.axes([0.15, 0.03, 0.65, 0.03])
    axc = plt.axes([0.15, 0.05, 0.65, 0.03])

    ssigm = Slider(axsigm, 'a', 0, 100, valinit=0, valstep=1)
    salpha = Slider(axalpha, 'a', -2, 0)
    sc = Slider(axc, 'a', 0, 3)

    def animate(i):
        x = np.abs(d_list[i])

        sigma = df.index[i]
        x[np.round(sigma * 10).astype(int):] = 0
        # y = np.abs(d_pred_list[i])
        alpha = df.at[sigma, 'alpha']
        salpha.set_val(alpha)
        c = df.at[sigma, 'c']
        sc.set_val(c)
        d_0 = df1.loc[sigma].iat[0]
        y = d_0 * np.exp(alpha * (k ** c))

        line1.set_data(k, x)
        line1.set_marker('.')
        line1.set_label('d')
        text2.set_text(('ошибка: ' + str(df.iat[i, 2])))
        text3.set_text(('$\sigma$: ' + str(sig[i])))
        line2.set_data(k, y, )
        line2.set_marker('.')
        return line1, line2, text2, text3,

    def update(val):
        alpha = salpha.val
        c = sc.val
        sigma = df.index[ssigm.val]
        d_0 = df1.loc[sigma].iat[0]
        y = d_0 * np.exp(alpha * (k ** c))
        line2.set_data(k, y, )
        line2.set_marker('.')

    for i in df.index:
        alpha = df.at[i, 'alpha']
        c = df.at[i, 'c']
        d = np.array(df1.loc[i])
        d_pred = d[0] \
                 * np.exp(alpha * (k ** c)) \
                 * (-1.) \
                 ** k
        d_list.append(d)
        d_pred_list.append(d_pred)

    ssigm.on_changed(animate)
    salpha.on_changed(update)
    sc.on_changed(update)

    plt.grid()
    plt.show()


@unused
def sigma_alpha_slider():
    df = pd.read_csv('gauss/sigma_alpha_err.csv'
                     , header=0
                     , index_col=['sigma']
                     )
    sig = df.index
    alph = df.loc[:, 'alpha']
    a = np.e
    f = -np.abs(1 / np.log2(a * sig)) + 1 / np.e
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 2), ylim=(-3, 0))
    ax.plot(sig, alph)
    line, = ax.plot(sig, f)
    axa = plt.axes([0.15, 0.01, 0.65, 0.03])
    axb = plt.axes([0.15, 0.03, 0.65, 0.03])
    axc = plt.axes([0.15, 0.06, 0.65, 0.03])

    sa = Slider(axa, 'a', 0, 10.0, valinit=np.e)
    sb = Slider(axb, 'b', 0, 10, valinit=2)
    sc = Slider(axc, 'c', 0, 2, valinit=1)

    def update(val):
        a = sa.val
        b = sb.val
        c = sc.val
        f = -np.abs(np.array(1 / log(a * sig ** c, b))) + 1 / np.e
        line.set_ydata(f)
        plt.draw()

    sa.on_changed(update)
    sb.on_changed(update)
    sc.on_changed(update)
    plt.show()


def fi_approx_fi_slider():
    fi_df = pd.read_csv('gauss/sigma_fi.csv'
                        , header=0
                        , index_col=['sigma']
                        )
    approx_fi_df = pd.read_csv('gauss/sigma_approx_fi2.csv'
                               , header=0
                               , index_col=['sigma']
                               )
    sigmas = fi_df.index.tolist()
    fi = fi_df.loc[0.1]

    approx_fi = approx_fi_df.loc[0.1]
    x = np.linspace(BOUND_A, BOUND_B, 1000)
    fig = plt.figure()
    ax = plt.axes(xlim=(BOUND_A, BOUND_B), ylim=(-1.2, 1.2))
    major_xticks, minor_xticks = np.arange(BOUND_A, BOUND_B + 1, 1), np.arange(BOUND_A - 0.5, BOUND_B - 0.5, 1)
    major_yticks, minor_yticks = np.arange(-1.2, 1.4, 0.2), np.arange(-1.1, 1.3, 0.2)
    ax.set_xticks(major_xticks)
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(major_yticks)
    ax.set_yticks(minor_yticks, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.8)

    line1, = plt.plot(x, fi, label='$\phi$')
    line2, = plt.plot(x, approx_fi, label='$\hat{\phi}$')
    # d1, = plt.
    # plt.scatter(x_sc, fi.loc[::50], marker = '.')
    plt.legend()

    axa = plt.axes([0.15, 0.01, 0.65, 0.03])
    axd = plt.axes([0.15, 0.03, 0.65, 0.03])
    # axb = plt.axes([0.15, 0.04, 0.65, 0.03])
    # axc = plt.axes([0.15, 0.06, 0.65, 0.03])

    sa = Slider(axa, 'a', 0, 100, valinit=0, valstep=1)
    sd = Slider(axd, 'd', 0, 100, valinit=0, valstep=1)
    # sb = Slider(axb, 'alpha', -3, 0, valinit=0, valstep=0.00001)
    # sc = Slider(axc, 'c', 0, 3, valinit=0, valstep=0.00001)

    df = pd.read_csv('gauss/sigma_alpha_c_err.csv'
                     , header=0
                     , index_col=['sigma']
                     )

    def update(val):
        a = sa.val
        # d = sd.val
        # sb.set_val(df.at[sigmas[a],'alpha'])
        # sc.set_val(df.at[sigmas[a],'c'])
        # print(a)
        # print(len(sigmas))
        # f = np.abs(1 / (np.log2(1.9 * sigmas[a]) + 0.1)) + 1
        appr = approx_fi_df.loc[sigmas[a]]
        fi = fi_df.loc[sigmas[a]]

        line1.set_data(x, fi)
        line2.set_data(x, appr)
        # plt.legend()
        plt.title(f'$\sigma = {sigmas[a]}$')
        plt.draw()

    kk = np.arange(0, RMAX + 1)

    def update2(val):
        sigma_ind = sa.val
        sigma = sigmas[sigma_ind]
        fi = fi_df.loc[sigma]
        line1.set_data(x, fi)
        plt.title(f'$\sigma = {sigma}$')
        true_d = pd.read_csv('gauss/d_native.csv'
                             , header=0
                             , index_col=['sigma']
                             ).loc[sigma]
        ind = sd.val

        alpha, c, _ = df.loc[sigma]
        d_0 = true_d.iat[0]

        d = d_0 \
            * np.exp(alpha *
                     (kk ** c)) \
            * (-1.) ** kk

        d[:ind + 1] = true_d.iloc[:ind + 1]
        # d[ind:] = true_d.iloc[ind:]
        # d[np.round(sigma*10).astype(int):]=0

        approx_fi = []

        for xx in x:
            fi_val = 0
            for k in range(-RMAX, RMAX + 1):
                fi_val += d[abs(k)] * exp(-((xx - k) * (xx - k)) / (2 * sigma * sigma))

            fi_val *= 1 / C(sigma)
            approx_fi.append(fi_val)
        # print(len(x),len(approx_fi))
        line2.set_data(x, approx_fi)

        plt.draw()

    sa.on_changed(update2)
    sd.on_changed(update2)
    # sb.on_changed(update2)
    # sc.on_changed(update2)
    plt.show()


def alpha_sigma_plot():
    df = pd.read_csv('gauss/sigma_alpha_c_err.csv'
                     , header=0
                     # , index_col=['sigma']
                     )
    # plt.plot('sigma','alpha',data=df.iloc[40:,:])
    fig = plt.figure()
    # ax = plt.axes(xlim=(0.1, 2.1),ylim=(-3, 0))
    ax = plt.axes(xlim=(0.1, 2.1), ylim=(0, 5))
    # plt.plot('sigma', 'alpha', '', data=df)
    plt.plot('sigma', 'c', '', data=df)
    a = 1.9
    sig = np.array(df.loc[:, 'sigma'])
    ax.set_xticks(sig, minor=True)
    # ax.set_xticks(sig[::10])
    plt.grid(which='minor', alpha=0.4)
    plt.grid(which='major', alpha=0.8)
    # f = np.abs(1 / (np.log2(a * sig) + 0.1)) + 1
    # plt.plot(sig[40:], f[40:])
    # plt.plot(sig, f)
    plt.show()


if __name__ == '__main__':
    # calculate_alphas2()
    sigma_d_slider()
    # calculate_fi()
    fi_approx_fi_slider()
    # alpha_sigma_plot()
