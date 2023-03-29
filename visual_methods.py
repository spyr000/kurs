import matplotlib.collections
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from utils import visual, Parameters, RMAX
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
    # vline, = ax.plot([], [], lw=0.5)
    error_text = ax.text(0.55, 0.85, '', transform=ax.transAxes)  # текст для вывода ошибки
    sigma_text = ax.text(0.55, 0.75, '', transform=ax.transAxes)  # текст для вывода значения sigma

    d_list = []
    d_pred_list = []

    axc = plt.axes([0.15, 0.05, 0.65, 0.03])  # ось ползунка для параметра c
    axalpha = plt.axes([0.15, 0.03, 0.65, 0.03])  # ось ползунка для параметра alpha
    axsigm = plt.axes([0.15, 0.01, 0.65, 0.03])  # ось ползунка для параметра sigma
    # axcc = plt.axes([0.01, 0.05, 0.0225, 0.63])
    # scc = Slider(
    #     ax=axcc,
    #     label=parameters.file_coeffs[2],
    #     valmin=0,
    #     valmax=3,
    #     orientation="vertical"
    # )

    sc = Slider(axc, parameters.file_coeffs[1], 0, 3)  # ползунок для параметра c
    salpha = Slider(axalpha, parameters.file_coeffs[0], -2, 0)  # ползунок для параметра alpha
    ssigm = Slider(axsigm, '$\sigma$', 0, 100, valinit=0, valstep=1)  # ползунок для параметра sigma

    for i in coeff_df.index:  # проходимся по всем сигмам
        d = np.array(d_df.loc[i])
        d_list.append(d)  # записываем в список с коэффициентами d для разных сигм

    def update_sigma(i):
        true_d = np.abs(d_list[i])  # считываем настоящие коэффициенты d
        sigma = coeff_df.index[i]
        true_d[np.round(sigma * 10).astype(int):] = 0  # обнуляем маловлияющие коэффициенты d
        # alpha = coeff_df.at[sigma, parameters.file_coeffs[0]]  # считываем значение параметра alpha
        # c = coeff_df.at[sigma, parameters.file_coeffs[1]]  # считываем значение параметра с
        coeffs = coeff_df.loc[sigma]

        salpha.set_val(coeffs.iat[0])  # устанавливаем значение параметра alpha
        sc.set_val(coeffs.iat[1])  # устанавливаем значение ползунка с
        # scc.set_val(coeffs.iat[2])
        d_0 = d_df.loc[sigma].iat[0]
        pred_d = parameters.approx_func(d_0, coeffs[:-1], False)  # пересчитываем приближенные коэффициенты d
        # pred_d = d_0 * np.exp(alpha * (k ** c))  # пересчитываем приближенные коэффициенты d

        true_d_line.set_data(k, true_d)
        # обновляем координаты линии настоящих коэффициетов d
        true_d_line.set_marker('.')
        true_d_line.set_label('d')
        # обновляем координаты линии приближенных коэффициетов d
        pred_d_line.set_data(k, pred_d, )
        pred_d_line.set_marker('.')
        # xv = np.round(sigma * 6).astype(int)
        # xv = d_0
        # vline.set_data(np.linspace(parameters.bounds[0],parameters.bounds[1],10),np.full(10,xv))
        # print(sigma, xv,np.argmin(true_d[:xv+1]))

        error_text.set_text(('ошибка: ' + str(coeff_df.iat[i, -1])))  # обновляем текст для вывода ошибки
        sigma_text.set_text(('$\sigma$: ' + str(sigmas[i])))  # обновляем текст для вывода значения sigma

        return true_d_line, pred_d_line, error_text, sigma_text,

    def update_params(val):
        alpha = salpha.val
        c = sc.val
        # cc = scc.val
        sigma = coeff_df.index[ssigm.val]
        d_0 = d_df.loc[sigma].iat[0]
        # pred_d = d_0 * np.exp(alpha * (k ** c))
        pred_d = parameters.approx_func(d_0, [alpha, c] + coeff_df.iloc[ssigm.val, 2:-1].tolist(), False)
        # pred_d = parameters.approx_func(d_0, [alpha, c, cc] + coeff_df.iloc[ssigm.val,3:-1].tolist(), False)
        # pred_d = parameters.approx_func(d_0, alpha, c)
        # обновляем координаты линии приближенных коэффициетов d
        pred_d_line.set_data(k, pred_d, )
        pred_d_line.set_marker('.')

    ssigm.on_changed(update_sigma)
    salpha.on_changed(update_params)
    sc.on_changed(update_params)
    # scc.on_changed(update_params)
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
    # alpha, c, _ = coeff_df.loc[0.1]
    coeffs = coeff_df.loc[0.1]
    d = parameters.approx_func(d_df.loc[0.1].iat[0], coeffs.iloc[:-1])

    approx_fi = parameters.phi_wave(x, 0.1, d=d)

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

        coeffs = coeff_df.loc[sigma].iloc[:-1]
        # coeffs[np.round(sigma * 10).astype(int):] = 0
        d_0 = true_d.iat[0]

        # d = d_0 \
        #     * np.exp(alpha *
        #              (kk ** c)) \
        #     * (-1.) ** kk
        d = parameters.approx_func(d_0, coeffs)

        d[:ind + 1] = true_d.iloc[
                      :ind + 1]  # заменяем ind первых приближенных коэффициентов d настоящими коэффициентами
        # d[ind:] = true_d.iloc[ind:]
        # d[np.round(sigma*10).astype(int):]=0


        approx_fi = parameters.phi_wave(x, sigma, d=d)

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
        plt.title('alpha')
    elif coeff_num == 0:
        ax = plt.axes(xlim=(0.1, 2.1), ylim=(0, 5))
        plt.title('С')
        plt.plot('sigma', 'c', '', data=df)
    else:
        ax = plt.axes(xlim=(0.1, 2.1), ylim=(0, df.iloc[-1].at['error']))
        plt.title('Ошибка')
        plt.plot('sigma', 'error', '', data=df)


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