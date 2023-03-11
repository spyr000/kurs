from math import exp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tqdm as tqdm
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


def get_y(sigma, x):
    a, b = 8.007502736216965, 0.8211001
    ln_c = -1.4524905467690978
    # y1 = deepcopy(y)
    # for i in tqdm(range(1)):
    #     y_ = np.log(y1)
    #     b_f_vals = np.array(a / (x + b) + ln_c).reshape(-1, 1)
    #     regr = LinearRegression()
    #     regr.fit(b_f_vals, y_)
    #     a *= regr.coef_[0]
    #     ln_c += regr.intercept_
    #     y1 = np.exp(a / (x + b) + ln_c)
    #     div = np.log(y) - ln_c
    #     x_pred = (a - b * (div)) / div
    #     n = np.mean(x_pred - x)
    #     if np.mean(np.abs((np.exp(a / (x + b - n) + ln_c) - y) / y)) < np.mean(np.abs((y1 - y) / y)):
    #         b -= n
    # alpha = np.exp(a / (sigma + b) + ln_c)
    # print(alpha)
    d = pd.read_csv('d_0-2.csv', header=0, index_col=['sigma']).loc[sigma][0]
    alpha = pd.read_csv('../sigma-alpha-norm_0-2.csv', header=0, index_col=['sigma']).loc[sigma].loc['alpha']
    # print(alpha)
    out = d * np.exp(-alpha * np.abs(x))
    return out

# @vectorize([(float64, float64, bool_)], target="parallel", nopython=True, cache=True)
def fi_wave(x: np.ndarray, sigma: float, flag: bool):
    def inner(x: np.ndarray, sigma: float, flag: bool):
        kk = np.arange(-RMAX, RMAX + 1)
        fi_val = 0
        if flag:
            d = d_list(sigma)
            for k in range(-RMAX, RMAX + 1):
                fi_val += d[abs(k)] * np.exp(-((x - k) * (x - k)) / (2 * sigma * sigma))
        else:
            d = get_y(sigma, kk)
            for k in range(-RMAX, RMAX + 1):
                fi_val += d[abs(k)] * np.exp(-((x - k) * (x - k)) / (2 * sigma * sigma))

        return 1 / C(sigma) * fi_val
    return np.vectorize(inner)(x, sigma, flag)

if __name__ == '__main__':
    sigma = 1
    x = np.linspace(BOUND_A, BOUND_B, 1000)
    y = fi_wave(x, sigma, True)
    y1 = fi_wave(x, sigma, False)
    # d = np.abs(d_list(sigma))
    # o = open('coefficients.csv', 'w')
    # for i in range(len(d)):
    #     o.write(f'{i},{d[i]:.3e}\n')
    # o.close()
    fig, ax = plt.subplots(figsize=(12/10*8, 10/12*8))
    plt.grid()
    plt.plot(x, y)
    plt.plot(x, y1)
    # plt.plot(d)
    plt.show()

if __name__ == '__main__':
    sigma = 1
    x = np.linspace(BOUND_A, BOUND_B, 1000)
    y = fi_wave(x, sigma, True)
    y1 = fi_wave(x, sigma, False)
    # d = np.abs(d_list(sigma))
    # o = open('coefficients.csv', 'w')
    # for i in range(len(d)):
    #     o.write(f'{i},{d[i]:.3e}\n')
    # o.close()
    fig, ax = plt.subplots(figsize=(12/10*8, 10/12*8))
    plt.grid()
    plt.plot(x, y)
    plt.plot(x, y1)
    # plt.plot(d)
    plt.show()



def write_d_to_csv():
    df_list = []
    k = np.linspace(0, 2, RMAX + 1)
    for sigma in k:
        df_list.append(np.abs(d_list(sigma)))
    df = pd.DataFrame(df_list, index=pd.Index(k, name='sigma'))
    df.to_csv('d_0-2.csv')


def write_coeffs_to_csv():
    d_df = pd.read_csv('d_0-2.csv', header=0, index_col=['sigma'])
    k = np.array(d_df.index)
    coeffs = []
    for sigma in k:
        d = d_df.loc[sigma]
        # print(d)
        err_f = []
        a_variations = np.arange(0.21, 1000, 0.01)
        for i in a_variations:
            error = np.linalg.norm(d[0] * np.exp(-i * np.abs(k)) - d)
            err_f.append(error)
        arr = np.array(err_f)
        min_err = arr.min()
        min_num = arr.argmin()
        coeffs.append([sigma, a_variations[min_num], min_err])
        # indexes.append(sigma)

    df = pd.DataFrame(np.array(coeffs), columns=['sigma', 'alpha', 'norm'])
    df.to_csv('sigma-alpha-norm_0-2.csv', index=False)


def get_a_sigma_coeffs():
    df = pd.read_csv('../sigma-alpha-norm_0-2.csv')
    ln = 8
    x = np.array(df['sigma'])[ln:].reshape(-1, 1)
    # x = x[x >= 0.2]
    y_first = np.array(df['alpha'])

    y = np.array(df['alpha'])[ln:]
    length = len(y)
    y_actual = y * np.ones(length).reshape(-1, 1)
    # mx = y.max()
    norm = []
    l = len(y_first[y_first <= 6])
    b_s = np.arange(0, 1000, 0.1)
    for b in b_s:
        y_pred = np.exp(b / (x + 1))
        norm.append(np.linalg.norm(y_actual - y_pred))
    out = b_s[6 + np.array(norm).argmin()]
    bias = np.max(np.exp(out / (x + 1)))
    return out, np.linalg.norm(out - bias)


# if __name__ == '__main__':
#     # write_d_to_csv()
#     # write_coeffs_to_csv()
#     df = pd.read_csv('sigma-alpha-norm_0-2.csv')
#     lplt = sns.lineplot(x='sigma', y='alpha', data=df, label='Альфа', color='r')
#     bias = df['norm'] * 100
#     sigma_cnt = len(df['sigma']) - 1
#     colors = mpl.colormaps['viridis']
#     x = df['sigma']
#     y = df['alpha']
#     delta_norm = bias / bias.max()
#     for i in range(sigma_cnt - 1):
#         lower, upper = y[i:i + 2] - bias[i:i + 2], y[i:i + 2] + bias[i:i + 2]
#         chunk = lplt.fill_between(x[i:i + 2], lower, upper, alpha=1,
#                                   color=colors(1 - delta_norm[i]))
#     a = 9
#     y1 = np.exp(a/(x+1+0.2))
#     # y1 = 1/np.log(x/a+1)
#     y1[y1>1000]=1000
#     # y1[y1 > 70] = 70
#     plt.plot(x, y1,color='orange')
#     # plt.plot(x,y1)
#     plt.grid()
#     plt.legend()
#     cbar = plt.colorbar(chunk, orientation='horizontal', label="Ошибка", shrink=.75)
#     plt.show()


if __name__ == '__main__':
    # write_coeffs_to_csv()
    df = pd.read_csv('../sigma-alpha-norm_0-2.csv', header=0, usecols=['sigma', 'alpha', 'norm'])
    ln = 8
    f, ax = plt.subplots()
    plt.grid()
    x = df['sigma'].iloc[ln:]
    y = df['alpha'].iloc[ln:]

    line_plt = sns.lineplot(x=x, y=y, label='Альфа', color=(1, 0, 0, 0.6))
    increase = 100
    bias = increase * df['norm'].iloc[ln:]
    sigma_cnt = len(x) - 1
    colors = mpl.colormaps['viridis']

    # bias_norm = bias / bias.max()
    # bias_norm = preprocessing.normalize([np.array(bias)])[0]
    chunk = None
    for i in range(sigma_cnt):
        chunk = ax.fill_between(x[i:i + 2], y[i:i + 2] - bias[i:i + 2], y[i:i + 2] + bias[i:i + 2], alpha=1,
                                color=colors(bias.iat[i]))

    # c, b, a = get_c_coeff(x, y)
    # c, b, a = 0.15, 0.75, 8
    # a = 7.286199689313037
    # b = 0.75
    # ln_c = -1.238298650175412
    # c, b, a = 0.2, 0.8211001, 8.007502736216965
    a, b = 8.007502736216965, 0.8211001
    ln_c = -1.4524905467690978
    # c, b, a = 0.37, 0.53, 6
    # a, b = 5.986258272067828, 0.6147663644049396
    # ln_c = -0.8175389212278102
    # a, b = 4.695062708606065, 0.47
    # ln_c = -0.34271782533516837
    y1 = deepcopy(y)
    for i in tqdm(range(1)):
        y_ = np.log(y1)
        b_f_vals = np.array(a / (x + b) + ln_c).reshape(-1, 1)
        regr = LinearRegression()
        regr.fit(b_f_vals, y_)
        a *= regr.coef_[0]
        ln_c += regr.intercept_
        y1 = np.exp(a / (x + b) + ln_c)
        div = np.log(y) - ln_c
        x_pred = (a - b * (div)) / div
        n = np.mean(x_pred - x)
        if np.mean(np.abs((np.exp(a / (x + b - n) + ln_c) - y) / y)) < np.mean(np.abs((y1 - y) / y)):
            b -= n
    y1 = np.exp(a / (x + b) + ln_c)
    mape = [np.mean(np.abs((y1[i:] - y[i:]) / y[i:])) for i in range(len(y))]
    print('a=', a, 'b=', b, 'ln_c=', ln_c)
    print(y)
    print(mape[0])
    sns.lineplot(x=x, y=y1, color=(0.2, 0.0, 1, 0.6), label='MAPE=' + str(mape[0]))

    cax, _ = cbar.make_axes(ax)
    ticks = np.linspace(0, bias.max(), increase)
    tick_labels = ticks / increase
    plt.legend()
    cb2 = cbar.ColorbarBase(cax, cmap='viridis', ticks=ticks, label='Ошибка')
    cb2.set_ticklabels(tick_labels.round(4))

    plt.show()
    plt.plot(mape)
    plt.grid()
    plt.show()


