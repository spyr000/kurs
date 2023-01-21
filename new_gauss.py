import math
from functools import partial
from math import exp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tqdm as tqdm
from matplotlib import animation
from numba import njit, vectorize, float64, bool_
import pandas as pd
import seaborn as sns
import matplotlib.colorbar as cbar
from sklearn import preprocessing
from matplotlib.widgets import Slider, Button
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from copy import deepcopy

RMAX = 100
BOUND_A, BOUND_B = -10, 10

decorator = partial(np.vectorize, excluded=['sigma'])


def customizable_vectorize(excluded=None):
    if excluded is None:
        excluded = []

    def inner_decorator(func):
        def wrapped(*args, **kwargs):
            return np.vectorize(func(*args, **kwargs), excluded=excluded)

        return wrapped

    return inner_decorator


def log(x, base):
    return np.vectorize(math.log)(x, base)


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
    # kk = np.arange(-RMAX, RMAX + 1)
    fi_val = 0
    # d = d_list(sigma)
    # print(d)
    d = pd.read_csv('d_native.csv'
                    , header=0
                    , index_col=['sigma']
                    )
    d = np.array(d.loc[sigma])

    for k in range(-RMAX, RMAX + 1):
        # print(d[abs(k)])
        fi_val += d[abs(k)] * exp(-((x - k) * (x - k)) / (2 * sigma * sigma))

    return 1 / C(sigma) * fi_val


def gradient_desc(sigma, alpha=0, lr=1e-4):
    d = pd.read_csv('d_native.csv'
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


def is_pos_def(x):
    return x[0, 0] > 0 and (x[0, 0] * x[1, 1] - x[0, 1] * x[1, 0]) > 0


def gradient_desc2(sigma, alpha=0, c=0, lr=1e-6, h_a=0.7, h_c=0.7):
    d = pd.read_csv('d_native.csv'
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
        # if np.any(np.isnan(d_pred)):
        #     print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        #     alpha = 0
        #     c = 0
        #     d_pred = d_0 * np.exp(alpha * (k ** c))
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
            alpha2_derivative = 1 / RMAX * \
                                np.sum(
                                    -2 * d_pred * (k ** (2 * c)) * d_0 * np.exp((k ** c) * alpha) \
                                    + 4 * (k ** (2 * c)) * (d_0 ** 2) * np.exp(2 * (k ** c) * alpha)
                                )
            alpha_c_derivative = 1 / RMAX * \
                                 np.sum(
                                     4 * (k ** (2 * c)) * alpha * (d_0 ** 2) * np.exp(2 * (k ** c) * alpha) * np.log(k) \
                                     - 2 * d_pred * (k ** c) * d_0 * np.log(k) * np.exp((k ** c) * alpha) \
                                     - 2 * d_pred * (k ** (2 * c)) * alpha * d_0 * np.log(k) * np.exp((k ** c) * alpha) \
                                     + 2 * (k ** c) * (d_0 ** 2) * np.exp(2 * (k ** c) * alpha) * np.log(k)
                                 )
            c2_derivative = 1 / RMAX * \
                            np.sum(
                                4 * (k ** (2 * c)) * (alpha ** 2) * (d_0 ** 2) * np.exp(2 * (k ** c) * alpha) * (
                                        np.log(k) ** 2) \
                                - 2 * d_pred * (k ** c) * alpha * d_0 * (np.log(k) ** 2) * np.exp((k ** c) * alpha) \
                                - 2 * d_pred * (k ** (2 * c)) * (alpha ** 2) * d_0 * (np.log(k) ** 2) * np.exp(
                                    (k ** c) * alpha) \
                                + 2 * (k ** c) * alpha * (d_0 ** 2) * np.exp(2 * (k ** c) * alpha) * (np.log(k) ** 2)
                            )

            hessian = np.array([
                [alpha2_derivative, alpha_c_derivative],
                [alpha_c_derivative, c2_derivative]
            ])
            # print(hessian)
            if is_pos_def(hessian):
                _hessian = np.linalg.inv(hessian)
                grad = np.array([[alpha_derivative], [c_derivative]])
                _d = (_hessian @ grad).flatten()
                alpha -= h_a * _d[0]
                c -= h_c * _d[1]
            else:
                alpha -= h_a * alpha_derivative
                c -= h_c * c_derivative
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
            print(f'\t{i}) альфа:', alpha, 'степень:', c, 'ошибка:', eps, 'сигма:', sigma)
            if i > max_iter:
                flag = False
            prev_eps = eps
            i += 1


    return alpha, c, eps, h_a, h_c

def newton_gauss(sigma, h_a, h_c, alpha=0, c=0, lr=1e-6):
    d = pd.read_csv('d_native.csv'
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
            alpha_derivative = -d_0 * (k ** c)\
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
                    _d = np.linalg.pinv(jacobian) @\
                         np.hstack((r := (d_pred - d).reshape(-1,1), r))
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

def calculate_alphas2():
    d = pd.read_csv('d_native.csv'
                    , header=0
                    , index_col=['sigma']
                    )
    sigmas = np.array(d.index)
    a = []
    alpha, c = 0, 0
    h_a, h_c = 0.00001, 0.00001
    for sigma in sigmas:
        alpha, c, err, h_a, h_c = newton_gauss(sigma, alpha=alpha, c=c, h_a=h_a, h_c=h_c)
        a.append([sigma, alpha, c, err])
    # print(a)
    df = pd.DataFrame(a, columns=['sigma', 'alpha', 'c', 'error'])
    df.set_index('sigma', inplace=True)
    print(df)
    df.to_csv('sigma_alpha_c_err2.csv')
    return df


def calculate_alphas():
    d = pd.read_csv('d_native.csv'
                    , header=0
                    , index_col=['sigma']
                    )
    sigmas = np.array(d.index)
    a = []
    for sigma in sigmas:
        alpha, err = gradient_desc2(sigma)
        a.append([sigma, alpha, err])
    # print(a)
    df = pd.DataFrame(a, columns=['sigma', 'alpha', 'error'])
    df.set_index('sigma', inplace=True)
    print(df)
    df.to_csv('sigma_alpha_err.csv')
    return df


@np.vectorize
def approx_fi_wave(x: np.ndarray, sigma: float):
    kk = np.arange(0, RMAX + 1)
    fi_val = 0
    # d = d_list(sigma)
    # print(d)
    d_0 = pd.read_csv('d_native.csv'
                      , header=0
                      , index_col=['sigma']
                      ).loc[sigma].iat[0]
    d = pd.read_csv('sigma_alpha_err.csv'
                    , header=0
                    , index_col=['sigma']
                    )

    alpha = np.array(d.at[sigma, 'alpha'])
    d = d_0 * np.exp(alpha * np.abs(kk)) * (-1.) ** kk

    for k in range(-RMAX, RMAX + 1):
        # print(d[abs(k)])
        fi_val += d[abs(k)] * exp(-((x - k) * (x - k)) / (2 * sigma * sigma))

    return 1 / C(sigma) * fi_val


@np.vectorize
def approx_fi_wave2(x: np.ndarray, sigma: float):
    kk = np.arange(0, RMAX + 1)
    fi_val = 0
    # d = d_list(sigma)
    # print(d)
    d_0 = pd.read_csv('d_native.csv'
                      , header=0
                      , index_col=['sigma']
                      ).loc[sigma].iat[0]
    d = pd.read_csv('sigma_alpha_c_err.csv'
                    , header=0
                    , index_col=['sigma']
                    )

    alpha = np.array(d.at[sigma, 'alpha'])
    c = np.array(d.at[sigma, 'c'])
    d = d_0 * np.exp(alpha * (kk ** c)) * (-1.) ** kk

    for k in range(-RMAX, RMAX + 1):
        # print(d[abs(k)])
        fi_val += d[abs(k)] * exp(-((x - k) * (x - k)) / (2 * sigma * sigma))

    return 1 / C(sigma) * fi_val


def write_d_to_csv():
    df_list = []
    k = np.linspace(0.1, 2.1, RMAX + 1)
    for sigma in k:
        df_list.append(d_list(sigma))
    df = pd.DataFrame(df_list, index=pd.Index(k, name='sigma'))
    df.to_csv('d_native.csv')

    # if __name__ == '__main__':
    #     sigma = 1
    #     a, b = 8.007502736216965, 0.8211001
    #     ln_c = -1.4524905467690978
    #
    #     # alpha = 0.6
    #
    #     kk = np.arange(RMAX + 1)
    #     y = np.abs(d_list(sigma))
    #     regr = LinearRegression()
    #     regr.fit(np.abs(kk).reshape(-1, 1), np.log(y).reshape(-1, 1))
    #     alpha = regr.coef_[0]
    #     print(alpha)
    #     intercept = regr.intercept_
    #
    #     y1 = np.exp(alpha * np.abs(kk))  # + intercept
    #
    #     plt.plot(y)
    #     plt.plot(y1)
    #     plt.show()


# if __name__ == '__main__':
#     df = pd.read_csv('sigma_alpha_c_err2.csv'
#                      , header=0
#                      , index_col=['sigma']
#                      )
#
#     fi_list, approx_fi_list = [], []
#
#     for sigma in tqdm(df.index):
#         x = np.linspace(BOUND_A, BOUND_B, 1000)
#         # y = fi_wave(x, sigma)
#         y1 = approx_fi_wave2(x, sigma)
#         # fi_list.append(y.tolist())
#         approx_fi_list.append(y1.tolist())
#
#     # df1 = pd.DataFrame(fi_list, index=df.index)
#     # df1.to_csv('sigma_fi.csv')
#     df1 = pd.DataFrame(approx_fi_list, index=df.index)
#     df1.to_csv('sigma_approx_fi2.csv')

    # df1 = pd.read_csv('sigma_fi.csv'
    #                   , header=0
    #                   , index_col='sigma'
    #                   )
    #
    # print(df1)
    # def makeArray(text):
    #     return np.fromstring(text, sep=' ')
    #
    # print(np.fromstring(df1.at[0.10,'fi'], sep=' '))
    # df.loc[:,'fi'] = df.loc[:,'fi'].apply(makeArray)
    # df.loc[:,'approx_fi'] = df.loc[:,'approx_fi'].apply(makeArray)
    # for i in range(y := df1.iat[0, 0]):
    #     print(i, y[i])
    # print(df1)
    # fig, ax = plt.subplots(figsize=(12 / 10 * 8, 10 / 12 * 8))
    # plt.grid()
    # plt.plot(x, y)
    # plt.plot(x, y1)
    # plt.show()
    # fig = plt.figure()


def write_d_abs_to_csv():
    df_list = []
    k = np.linspace(0, 2, RMAX + 1)
    for sigma in k:
        df_list.append(np.abs(d_list(sigma)))
    df = pd.DataFrame(df_list, index=pd.Index(k, name='sigma'))
    df.to_csv('d_0-2.csv')


def play_animation():
    df = pd.read_csv('sigma_alpha_err.csv'
                     , header=0
                     , index_col=['sigma']
                     )
    df1 = pd.read_csv('d_native.csv'
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


def play_animation2():
    df = pd.read_csv('sigma_alpha_c_err.csv'
                     , header=0
                     , index_col=['sigma']
                     )
    df1 = pd.read_csv('d_native.csv'
                      , header=0
                      , index_col=['sigma']
                      )
    sig = np.array(df.index)
    k = np.arange(0, RMAX + 1)
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 101), ylim=(-1, 1))
    line1, = ax.plot([], [], lw=2)
    line2, = ax.plot([], [], lw=0.5)
    text2 = ax.text(0.55, 0.85, '', transform=ax.transAxes)
    text3 = ax.text(0.55, 0.75, '', transform=ax.transAxes)

    def init():
        line1.set_data([], [])
        line2.set_data([], [])

        return line1, line2,

    xdata, ydata = [], []
    d_list = []
    d_pred_list = []

    def animate(i):
        x = d_list[i]
        y = d_pred_list[i]

        line1.set_data(k, x)
        line1.set_marker('.')
        line1.set_label('d')
        text2.set_text(('ошибка: ' + str(df.iat[i, 2])))
        # print(sig[i])
        text3.set_text(('$\sigma$: ' + str(sig[i])))
        line2.set_data(k, y, )
        line2.set_marker('.')
        plt.legend()
        return line1, line2, text2, text3,

    for i in df.index:
        alpha = df.at[i, 'alpha']
        c = df.at[i, 'c']
        d = np.array(df1.loc[i])
        d_pred = d[0] * np.exp(alpha * (k ** c)) * (-1.) ** k
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


# if __name__ == '__main__':
    # calculate_alphas2()
    # play_animation2()


    # d = pd.read_csv('d_native.csv'
    #                 , header=0
    #                 , index_col=['sigma']
    #                 )
    #
    # k = np.arange(0, RMAX + 1)
    # fig = plt.figure()
    # ax = plt.axes(xlim=(0, RMAX), ylim=(0, 3))
    # line1, = plt.plot(k, np.abs(d.loc[0.10]), label='$\phi$')
    # alpha = 0
    # line2, = plt.plot(k, d.loc[0.10].iat[0] * np.exp(-alpha * k ** 2), label='$\hat{\phi}$')
    #
    # axalpha = plt.axes([0.15, 0.01, 0.65, 0.03])
    # axsigma = plt.axes([0.15, 0.03, 0.65, 0.03])
    # axc = plt.axes([0.15, 0.06, 0.65, 0.03])
    #
    # salpha = Slider(axalpha, 'alpha', 0, 10, valinit=1, valstep=0.0001)
    # ssigma = Slider(axsigma, 'sigma', 0, 101, valinit=0, valstep=1)
    # sc = Slider(axc, 'c', 0, 4, valinit=1)
    #
    #
    # def update(val):
    #     alpha = salpha.val
    #     sigma = ssigma.val
    #     c = sc.val
    #
    #     line2.set_ydata(d.loc[d.index[sigma]].iat[0] * np.exp(-alpha * k ** c))
    #     line1.set_ydata(np.abs(d.loc[d.index[sigma]]))
    #     plt.draw()
    #
    #
    # salpha.on_changed(update)
    # ssigma.on_changed(update)
    # sc.on_changed(update)
    # plt.show()


def sigma_alpha_slider():
    df = pd.read_csv('sigma_alpha_err.csv'
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
    fi_df = pd.read_csv('sigma_fi.csv'
                        , header=0
                        , index_col=['sigma']
                        )
    approx_fi_df = pd.read_csv('sigma_approx_fi2.csv'
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
    # plt.scatter(x_sc, fi.loc[::50], marker = '.')
    plt.legend()

    axa = plt.axes([0.15, 0.01, 0.65, 0.03])
    sa = Slider(axa, 'a', 0, 101, valinit=0, valstep=1)

    def update(val):
        a = sa.val
        # b = sb.val
        # c = sc.val

        line1.set_data(x, fi_df.loc[sigmas[a]])
        line2.set_data(x, approx_fi_df.loc[sigmas[a]])
        # plt.legend()
        plt.title(f'$\sigma = {sigmas[a]}$')
        plt.draw()

    sa.on_changed(update)
    plt.show()


def alpha_sigma_plot():
    df = pd.read_csv('sigma_alpha_c_err.csv'
                     , header=0
                     # , index_col=['sigma']
                     )
    # plt.plot('sigma','alpha',data=df.iloc[40:,:])
    fig = plt.figure()
    # ax = plt.axes(xlim=(0.1, 2.1),ylim=(-0.3, 0))
    ax = plt.axes(xlim=(0.1, 2.1),ylim=(0, 5))
    print(df.dtypes)
    plt.plot('sigma', 'c','', data=df)
    a = 1.9
    sig = np.array(df.loc[:, 'sigma'])
    ax.set_xticks(sig,minor=True)
    ax.set_xticks(sig[::10])
    plt.grid(which='minor', alpha=0.4)
    plt.grid(which='major', alpha=0.8)
    f = np.abs(1 / (np.log2(a * sig) + 0.1)) + 1
    # plt.plot(sig[40:], f[40:])
    plt.plot(sig, f)
    plt.show()

if __name__ == '__main__':
    # pass
    # calculate_alphas2()
    # play_animation2()
    fi_approx_fi_slider()
    # alpha_sigma_plot()