import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from new import RMAX
from utils import visual, Parameters
from matplotlib.widgets import Slider


@visual
def sigma_d_slider(parameters: Parameters):
    coeff_df = pd.read_csv(parameters.parent_directory + '\\' + parameters.coeff_filename
                     , header=0
                     , index_col=['sigma']
                     )
    d_df = pd.read_csv(parameters.parent_directory + '\\' + parameters.d_filename
                      , header=0
                      , index_col=['sigma']
                      )
    sigmas = np.array(coeff_df.index)
    k = np.arange(0, RMAX + 1)  # набор координат по оси x
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 101), ylim=(0, 1))
    true_d_line, = ax.plot([], [], lw=2)  # линия для настоящих коэффициетов d
    pred_d_line, = ax.plot([], [], lw=0.5)  # линия для приближенных коэффициетов d
    error_text = ax.text(0.55, 0.85, '', transform=ax.transAxes)  # текст для вывода ошибки
    sigma_text = ax.text(0.55, 0.75, '', transform=ax.transAxes)  # текст для вывода значения sigma

    d_list = []
    d_pred_list = []

    axc = plt.axes([0.15, 0.05, 0.65, 0.03])  # ось ползунка для параметра c
    axalpha = plt.axes([0.15, 0.03, 0.65, 0.03])  # ось ползунка для параметра alpha
    axsigm = plt.axes([0.15, 0.01, 0.65, 0.03])  # ось ползунка для параметра sigma

    sc = Slider(axc, 'с', 0, 3)  # ползунок для параметра c
    salpha = Slider(axalpha, 'a', -2, 0)  # ползунок для параметра alpha
    ssigm = Slider(axsigm, '$\sigma$', 0, 100, valinit=0, valstep=1)  # ползунок для параметра sigma

    for i in coeff_df.index:  # проходимся по всем сигмам
        alpha = coeff_df.at[i, 'alpha']  # считываем параметр alpha
        c = coeff_df.at[i, 'c']  # считываем параметр c
        d = np.array(d_df.loc[i])  # считываем коэффициенты d
        d_pred = d[0] \
                 * np.exp(alpha * (k ** c)) \
                 * (-1.) \
                 ** k
        d_list.append(d)  # записываем в список с коэффициентами d для разных сигм
        d_pred_list.append(d_pred)  # записываем в список с приближенными коэффициентами d для разных сигм

    def update_sigma(i):
        true_d = np.abs(d_list[i])  # считываем настоящие коэффициенты d
        sigma = coeff_df.index[i]
        true_d[np.round(sigma * 10).astype(int):] = 0  # обнуляем маловлияющие коэффициенты d
        alpha = coeff_df.at[sigma, 'alpha']  # считываем значение параметра alpha
        salpha.set_val(alpha)  # устанавливаем значение параметра alpha
        c = coeff_df.at[sigma, 'c']  # считываем значение параметра с
        sc.set_val(c)  # устанавливаем значение ползунка с
        d_0 = d_df.loc[sigma].iat[0]
        pred_d = d_0 * np.exp(alpha * (k ** c))  # пересчитываем приближенные коэффициенты d

        true_d_line.set_data(k, true_d)
        # обновляем координаты линии настоящих коэффициетов d
        true_d_line.set_marker('.')
        true_d_line.set_label('d')
        # обновляем координаты линии приближенных коэффициетов d
        pred_d_line.set_data(k, pred_d, )
        pred_d_line.set_marker('.')

        error_text.set_text(('ошибка: ' + str(coeff_df.iat[i, 2])))  # обновляем текст для вывода ошибки
        sigma_text.set_text(('$\sigma$: ' + str(sigmas[i])))  # обновляем текст для вывода значения sigma

        return true_d_line, pred_d_line, error_text, sigma_text,

    def update_params(val):
        alpha = salpha.val
        c = sc.val
        sigma = coeff_df.index[ssigm.val]
        d_0 = d_df.loc[sigma].iat[0]
        pred_d = d_0 * np.exp(alpha * (k ** c))
        # обновляем координаты линии приближенных коэффициетов d
        pred_d_line.set_data(k, pred_d, )
        pred_d_line.set_marker('.')

    ssigm.on_changed(update_sigma)
    salpha.on_changed(update_params)
    sc.on_changed(update_params)
    plt.grid()
    plt.show()


@visual
def fi_approx_fi_slider(parameters: Parameters):
    fi_df = pd.read_csv(parameters.parent_directory + '\\' + parameters.phi_vals_filename
                        , header=0
                        , index_col=['sigma']
                        )

    coeff_df = pd.read_csv(parameters.parent_directory + '\\' + parameters.coeff_filename
                     , header=0
                     , index_col=['sigma']
                     )
    d_df = pd.read_csv(parameters.parent_directory + '\\' + parameters.d_filename
                       , header=0
                       , index_col=['sigma']
                       )

    sigmas = coeff_df.index.tolist()
    fi = fi_df.loc[0.1]

    x = np.linspace(parameters.bounds[0], parameters.bounds[1], 1000)

    fig = plt.figure()
    ax = plt.axes(xlim=(parameters.bounds[0], parameters.bounds[1]), ylim=(-1.2, 1.2))
    major_xticks, minor_xticks = np.arange(parameters.bounds[0], parameters.bounds[1] + 1, 1),\
                                 np.arange(parameters.bounds[0] - 0.5, parameters.bounds[1] - 0.5, 1)
    major_yticks, minor_yticks = np.arange(-1.2, 1.4, 0.2), np.arange(-1.1, 1.3, 0.2)
    ax.set_xticks(major_xticks)
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(major_yticks)
    ax.set_yticks(minor_yticks, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.8)

    kk = np.arange(0, RMAX + 1)  # значения номеров коэффициентов d соответствующие координате x графика
    alpha, c, _ = coeff_df.loc[0.1]
    d = d_df.loc[0.1].iat[0] \
        * np.exp(alpha *
                 (kk ** c)) \
        * (-1.) ** kk

    approx_fi = parameters.phi_wave(0.1, x, d=d)

    true_fi_line, = plt.plot(x, fi, label='$\phi$')  # линия для настоящей узловой функции fi
    line2, = plt.plot(x, approx_fi, label='$\hat{\phi}$')  # линия для приближенной узловой функции fi
    plt.legend()

    axsigma = plt.axes([0.15, 0.01, 0.65, 0.03])  # ось ползунка для параметра sigma
    axindex = plt.axes([0.15, 0.03, 0.65, 0.03])  # ось ползунка для параметра index

    ssigma = Slider(axsigma, '$\sigma$', 0, 100, valinit=0, valstep=1)  # ползунок для параметра sigma
    sindex = Slider(axindex, 'index', 0, 100, valinit=0, valstep=1)  # ползунок для параметра index

    def update(val):
        sigma_ind = ssigma.val  # считываем значение индекса sigma из значения ползунка
        sigma = sigmas[sigma_ind]
        fi = fi_df.loc[sigma]  # считываем значения настоящей узловой функции fi
        true_fi_line.set_data(x, fi)  # обновляем координаты линии настоящей узловой функции fi
        plt.title(f'$\sigma = {sigma}$')
        true_d = d_df.loc[sigma]
        ind = sindex.val  # считываем значение индекса из значения ползунка

        alpha, c, _ = coeff_df.loc[sigma]
        d_0 = true_d.iat[0]

        d = d_0 \
            * np.exp(alpha *
                     (kk ** c)) \
            * (-1.) ** kk

        d[:ind + 1] = true_d.iloc[
                      :ind + 1]  # заменяем ind первых приближенных коэффициентов d настоящими коэффициентами
        # d[ind:] = true_d.iloc[ind:]
        # d[np.round(sigma*10).astype(int):]=0


        approx_fi = parameters.phi_wave(sigma, x, d=d)

        line2.set_data(x, approx_fi)
        plt.draw()

    ssigma.on_changed(update)
    sindex.on_changed(update)
    plt.show()


@visual
def alpha_sigma_plot(parameters: Parameters, coeff_num=0):
    df = pd.read_csv(parameters.parent_directory + '\\' + parameters.coeff_filename
                     , header=0
                     )
    # plt.plot('sigma','alpha',data=df.iloc[40:,:])
    fig = plt.figure()
    if coeff_num < 1:
        ax = plt.axes(xlim=(0.1, 2.1),ylim=(-3, 0))
        plt.plot('sigma', 'alpha', '', data=df)
    else:
        ax = plt.axes(xlim=(0.1, 2.1), ylim=(0, 5))
        plt.plot('sigma', 'c', '', data=df)


    sig = np.array(df.loc[:, 'sigma'])
    ax.set_xticks(sig, minor=True)
    # ax.set_xticks(sig[::10])
    plt.grid(which='minor', alpha=0.4)
    plt.grid(which='major', alpha=0.8)
    # a = 1.9
    # f = np.abs(1 / (np.log2(a * sig) + 0.1)) + 1
    # plt.plot(sig[40:], f[40:])
    # plt.plot(sig, f)
    plt.show()