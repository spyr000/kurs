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

from new_gauss import RMAX, C
from utils import deprecated, unused, utility, moved


@deprecated
def play_animation():
    df = pd.read_csv('../gauss_resources/sigma_alpha_err.csv'
                     , header=0
                     , index_col=['sigma']
                     )
    df1 = pd.read_csv('../gauss_resources/d_native.csv'
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

    anim = animation.FuncAnimation(fig, animate,
                                   frames=101,
                                   interval=200,
                                   blit=True)
    plt.grid()
    plt.show()
    # anim.save('gif.gif',writer='imagemagick')


@deprecated
@np.vectorize
def approx_fi_wave(x: np.ndarray, sigma: float):
    '''функция расчета приближенного значения коэфициентов d'''
    kk = np.arange(0, RMAX + 1)  # список номеров коэффициентов
    fi_val = 0
    d_0 = pd.read_csv('gauss_resources/d_native.csv'
                      , header=0
                      , index_col=['sigma']
                      ).loc[sigma].iat[0]
    d = pd.read_csv('gauss_resources/sigma_alpha_err.csv'
                    , header=0
                    , index_col=['sigma']
                    )

    alpha = np.array(d.at[sigma, 'alpha'])
    d = d_0 * np.exp(alpha * np.abs(kk)) * (-1.) ** kk

    for k in range(-RMAX, RMAX + 1):
        fi_val += d[abs(k)] * exp(-((x - k) * (x - k)) / (2 * sigma * sigma))

    return 1 / C(sigma) * fi_val

@deprecated
def calculate_alphas():
    d = pd.read_csv('gauss_resources/d_native.csv'
                    , header=0
                    , index_col=['sigma']
                    )
    sigmas = np.array(d.index)
    a = []
    for sigma in sigmas:
        alpha, err = newton(sigma)
        a.append([sigma, alpha, err])
    df = pd.DataFrame(a, columns=['sigma', 'alpha', 'error'])
    df.set_index('sigma', inplace=True)
    print(df)
    df.to_csv('gauss_resources/sigma_alpha_err.csv')
    return df
