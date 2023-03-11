import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from new_gauss import BOUND_B, BOUND_A
from new_gauss import phi_wave, d_G_k
from calc_methods import approx_fi_wave2
from utils import unused, log, utility, RMAX
from functools import partial
from tqdm import tqdm


decorator = partial(np.vectorize, excluded=['sigma'])

@unused
@utility
def approx_func(sigma, alpha, c, k=np.arange(0, RMAX + 1)):
    return d_G_k(0, sigma) \
           * np.exp(alpha * (k ** c)) \
           * (-1.) \
           ** k

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


@unused
def sigma_alpha_slider():
    df = pd.read_csv('gauss_resources/sigma_alpha_err.csv'
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

@unused
def calculate_fi():
    df = pd.read_csv('gauss_resources/sigma_alpha_c_err.csv'
                     , header=0
                     , index_col=['sigma']
                     )

    fi_list, approx_fi_list = [], []
    x = np.linspace(BOUND_A, BOUND_B, 1000)

    for sigma in tqdm(df.index):
        y = phi_wave(x, sigma)
        y1 = approx_fi_wave2(x, sigma)
        # fi_list.append(y.tolist())
        approx_fi_list.append(y1.tolist())

    # df1 = pd.DataFrame(fi_list, index=df.index)
    # df1.to_csv('gauss_resources/sigma_fi.csv')
    df1 = pd.DataFrame(approx_fi_list, index=df.index)
    df1.to_csv('gauss_resources/sigma_approx_fi2.csv')
