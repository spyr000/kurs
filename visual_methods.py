import matplotlib.collections
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# import warnings
# warnings.filterwarnings("error")

from scipy.optimize import minimize

from MseConstruct import get_mse, mse



from utils import visual, Parameters, RMAX, target_approx_func
from matplotlib.widgets import Slider


def graphic(parameters: Parameters):
    d_df = pd.read_csv(parameters.parent_directory + '\\' + parameters.d_filename
                       , header=0
                       , index_col=['sigma']
                       )
    sigmas = np.array(d_df.index)
    k = np.arange(0, RMAX + 1)  # набор координат по оси x
    k_ = k
    k_[k_ == 0] = 1

    res_true = []
    res_pred = []
    for sigma in sigmas:
        true_d = np.abs(d_df[sigma])
        alpha_k = (np.log(RMAX - k - 2 * sigma * sigma * np.log(true_d / true_d[0])) - np.log(2 * sigma * sigma)) / k_
        pred_alpha_k = 1 / (k_ * 1 / 5 * sigma)
        res_true.append(alpha_k)
        res_pred.append(pred_alpha_k)

    plt.plot()


def calculate_hyperbola(parameters: Parameters):
    d_df = pd.read_csv(parameters.parent_directory + '\\' + parameters.d_filename
                       , header=0
                       , index_col=['sigma']
                       )
    sigmas = np.array(d_df.index)
    k = np.arange(0, RMAX + 1)  # набор координат по оси x
    fig = plt.figure()
    # ax = plt.axes(xlim=(0, 101), ylim=(-200, 0))
    ax = plt.axes(xlim=(0, 101), ylim=(0, 1))
    true_d_line, = ax.plot([], [], lw=2)  # линия для настоящих коэффициетов d
    pred_d_line, = ax.plot([], [], lw=0.5)  # линия для приближенных коэффициетов d
    # vline, = ax.plot([], [], lw=0.5)
    sigma_text = ax.text(0.55, 0.75, '', transform=ax.transAxes)  # текст для вывода значения sigma
    error_text = ax.text(0.55, 0.85, '', transform=ax.transAxes)  # текст для вывода ошибки
    axsigm = plt.axes([0.15, 0.01, 0.65, 0.03])  # ось ползунка для параметра sigma
    ssigm = Slider(axsigm, '$\sigma$', 0, 100, valinit=0, valstep=1)  # ползунок для параметра sigma

    vals = {}
    errors = {}
    for sigma in sigmas:
        k = np.arange(0, RMAX + 1)
        d = np.abs(np.array(d_df.loc[sigma]))

        true_vals = np.log(d / d[0]) + k / (s:=2*sigma*sigma)

        count_func = lambda x: RMAX/s * (1 - np.exp(-x[0]*k))

        result = minimize(get_mse(true_vals, count_func),
                          x0=np.array([1, 1]), method="nelder-mead", tol=10e-32)
        print(result)
        coeffs = result.x.tolist()

        vals.update({sigma: coeffs})
        errors.update({sigma : mse(true_vals,count_func(coeffs))})

    def update_sigma(i):
        true_d = np.abs(d_df.iloc[i])  # считываем настоящие коэффициенты d
        sigma = d_df.index[i]
        ab = vals[sigma]
        # pred_d_line.set_data(k, np.log(true_d / true_d[0]) + k / (s:=2*sigma*sigma))
        true_d_line.set_data(k, count_func(ab))


        true_d_line.set_marker('.')
        true_d_line.set_label('d')
        sigma_text.set_text(('$\sigma$: ' + str(sigmas[i])))  # обновляем текст для вывода значения sigma
        error_text.set_text(('ошибка: ' + str(errors[sigma])))  # обновляем текст для вывода ошибки

    ssigm.on_changed(update_sigma)
    plt.grid()
    plt.show()

def check_ratio2(parameters: Parameters):
    d_df = pd.read_csv(parameters.parent_directory + '\\' + parameters.d_filename
                       , header=0
                       , index_col=['sigma']
                       )
    sigmas = np.array(d_df.index)
    k = np.arange(0, RMAX + 1)  # набор координат по оси x
    ax = plt.axes(xlim=(0, 3), ylim=(-10, 0))
    true_d_line, = ax.plot([], [], lw=2)  # линия для настоящих коэффициетов d
    pred_d_line, = ax.plot([], [], lw=0.5)  # линия для приближенных коэффициетов d
    sigma_text = ax.text(0.55, 0.75, '', transform=ax.transAxes)  # текст для вывода значения sigma
    error_text = ax.text(0.55, 0.85, '', transform=ax.transAxes)  # текст для вывода ошибки
    d_list = []
    axsigm = plt.axes([0.15, 0.01, 0.65, 0.03])  # ось ползунка для параметра sigma
    ssigm = Slider(axsigm, '$\sigma$', 0, 100, valinit=0, valstep=1)  # ползунок для параметра sigma

    for i in d_df.index:  # проходимся по всем сигмам
        d = np.array(d_df.loc[i])
        d_list.append(d)  # записываем в список с коэффициентами d для разных сигм

    vals = {}
    def update_sigma(i):
        true_d = np.abs(d_list[i])  # считываем настоящие коэффициенты d
        sigma = d_df.index[i]

        k_ = k.copy()
        k_[k_ == 0] = 1
        log1 = None
        try:
            log = np.log(true_d / true_d[0])
        except RuntimeWarning:
            print(true_d[0], sigma)
        s = 2 * sigma * sigma
        true_d = np.log(-(np.log(true_d / true_d[-1]) + (k - RMAX)/s) / (np.log(true_d[-1]/true_d[0]) + RMAX/s))
        vals.update({sigma:true_d[1]})
        true_d_line.set_data(k, true_d)
        # print(alpha_k[1], 1 / (2 * sigma * sigma + 1))
        # param = 11
        # pred = (log[-1]+RMAX/s)  * (1 - np.exp(-k/(sigma*sigma)))
        # pred_d_line.set_data(k, pred )  # 1 - e
        # print(val)
        # обновляем координаты линии настоящих коэффициетов d
        true_d_line.set_marker('.')
        true_d_line.set_label('d')
        sigma_text.set_text(('$\sigma$: ' + str(sigmas[i])))  # обновляем текст для вывода значения sigma
        # error_text.set_text(('ошибка: ' + str(mse(true_d,pred))))  # обновляем текст для вывода ошибки

    ssigm.on_changed(update_sigma)
    plt.grid()
    plt.show()
    vals_list = []
    for sigma in sigmas:
        try:
            vals_list.append(vals[sigma])
        except KeyError:
            vals_list.append(0)
    plt.plot(sigmas,vals_list)
    plt.plot(sigmas,-1/sigmas)
    plt.show()

def check_ratio(parameters: Parameters):
    d_df = pd.read_csv(parameters.parent_directory + '\\' + parameters.d_filename
                       , header=0
                       , index_col=['sigma']
                       )
    sigmas = np.array(d_df.index)
    k = np.arange(0, RMAX + 1)  # набор координат по оси x
    fig = plt.figure()
    # ax = plt.axes(xlim=(0, 101), ylim=(-200, 0))
    ax = plt.axes(xlim=(0, 101), ylim=(0, 1))
    true_d_line, = ax.plot([], [], lw=2)  # линия для настоящих коэффициетов d
    pred_d_line, = ax.plot([], [], lw=0.5)  # линия для приближенных коэффициетов d
    # vline, = ax.plot([], [], lw=0.5)
    sigma_text = ax.text(0.55, 0.75, '', transform=ax.transAxes)  # текст для вывода значения sigma
    error_text = ax.text(0.55, 0.85, '', transform=ax.transAxes)  # текст для вывода ошибки
    d_list = []
    axsigm = plt.axes([0.15, 0.01, 0.65, 0.03])  # ось ползунка для параметра sigma
    ssigm = Slider(axsigm, '$\sigma$', 0, 100, valinit=0, valstep=1)  # ползунок для параметра sigma
    axparam = plt.axes([0.15, 0.04, 0.65, 0.03])  # ось ползунка для параметра sigma
    sparam = Slider(axparam, 'param', 9, 13, valinit=11, valstep=0.01)  # ползунок для параметра sigma

    for i in d_df.index:  # проходимся по всем сигмам
        d = np.array(d_df.loc[i])
        d_list.append(d)  # записываем в список с коэффициентами d для разных сигм

    arcth = lambda x: 0.5 * np.log((x + 1) / (x - 1))
    ones = np.ones(k.shape)
    def update_sigma(i):
        true_d = np.abs(d_list[i])  # считываем настоящие коэффициенты d
        sigma = d_df.index[i]
        param = sparam.val

        k_ = k.copy()
        k_[k_ == 0] = 1
        log = None
        try:
            log = np.log(true_d / true_d[0])
        except RuntimeWarning:
            print(true_d[0], sigma)
        # alpha_k = np.arctan(np.pi * sigma * sigma * log + np.pi * k / 2) / k_
        # alpha_k = (np.log(RMAX - k - 2*sigma*sigma*np.log(true_d/true_d[0])) - np.log(2*sigma*sigma)) / k_
        alpha_k = arcth((log + k/(s:=2*sigma*sigma)) / (RMAX/s + log[-1] - log[0]))  # / k_
        print((log + k/(s:=2*sigma*sigma)) / (RMAX/s + log[-1] - log[0]))
        # true_d_line.set_data(k, val:=(np.log(true_d / true_d[0]) + k/(2 * sigma**2)))
        # print(alpha_k)
        # true_d_line.set_data(k, alpha_k)
        print(alpha_k)
        # print(alpha_k[1], 1 / (2 * sigma * sigma + 1))
        # param = 11
        # pred = ((2 * sigma * sigma * np.log(true_d[-1] / true_d[0])) / RMAX + 1) * (
        #         1 - (5*sigma / (2 * sigma * sigma)) ** (-k))
        # pred_d_line.set_data(k, pred )  # 1 - e
        # print(val)
        # обновляем координаты линии настоящих коэффициетов d
        true_d_line.set_marker('.')
        true_d_line.set_label('d')
        sigma_text.set_text(('$\sigma$: ' + str(sigmas[i])))  # обновляем текст для вывода значения sigma
        error_text.set_text(('ошибка: ' + str(mse(alpha_k,pred))))  # обновляем текст для вывода ошибки


    def update_param(i):
        sigma_ind = ssigm.val  # считываем значение индекса sigma из значения ползунка
        # print(sigma_ind)
        sigma = sigmas[sigma_ind]
        true_d = np.abs(d_list[sigma_ind])
        alpha_k = np.arctanh((2 * sigma * sigma * np.log(true_d / true_d[0]) + k) / RMAX)
        pred = ((2 * sigma * sigma * np.log(true_d[-1] / true_d[0])) / RMAX + 1) * (
                1 - (i / (2 * sigma * sigma)) ** (-k))
        pred_d_line.set_data(k, pred)
        error_text.set_text(('ошибка: ' + str(mse(alpha_k,pred))))  # обновляем текст для вывода ошибки

    ssigm.on_changed(update_sigma)
    sparam.on_changed(update_param)
    plt.grid()
    plt.show()


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
    ax = plt.axes(xlim=(0, 101), ylim=(-0.2, 2))
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

    # axca = plt.axes([0.01, 0.05, 0.0225, 0.63])
    # sca = Slider(
    #     ax=axca,
    #     label=parameters.file_coeffs[2],
    #     valmin=-1,
    #     valmax=1,
    #     orientation="vertical"
    # )
    # axcb = plt.axes([0.04, 0.05, 0.0225, 0.63])
    # scb = Slider(
    #     ax=axcb,
    #     label=parameters.file_coeffs[3],
    #     valmin=-1,
    #     valmax=1,
    #     orientation="vertical"
    # )
    # axcc = plt.axes([0.07, 0.05, 0.0225, 0.63])
    # scc = Slider(
    #     ax=axcc,
    #     label=parameters.file_coeffs[4],
    #     valmin=-2,
    #     valmax=4,
    #     orientation="vertical"
    # )

    sc = Slider(axc, parameters.file_coeffs[1], 0, 50, valstep=0.01)  # ползунок для параметра c
    salpha = Slider(axalpha, parameters.file_coeffs[0], 0, 400, valstep=0.1)  # ползунок для параметра alpha
    ssigm = Slider(axsigm, '$\sigma$', 0, 100, valinit=0, valstep=1)  # ползунок для параметра sigma

    for i in coeff_df.index:  # проходимся по всем сигмам
        d = np.array(d_df.loc[i])
        d_list.append(d)  # записываем в список с коэффициентами d для разных сигм

    def update_sigma(i):
        true_d = np.abs(d_list[i])  # считываем настоящие коэффициенты d
        sigma = coeff_df.index[i]
        # true_d[np.round(sigma * 10).astype(int):] = 0  # обнуляем маловлияющие коэффициенты d
        # alpha = coeff_df.at[sigma, parameters.file_coeffs[0]]  # считываем значение параметра alpha
        # c = coeff_df.at[sigma, parameters.file_coeffs[1]]  # считываем значение параметра с
        coeffs = coeff_df.loc[sigma]

        salpha.set_val(coeffs.iat[0])  # устанавливаем значение параметра alpha
        sc.set_val(coeffs.iat[1])  # устанавливаем значение ползунка с
        # sca.set_val(coeffs.iat[2])
        # scb.set_val(coeffs.iat[3])
        # scc.set_val(coeffs.iat[4])
        # d_0 = d_df.loc[sigma].iat[0]
        alpha_k = np.arctan(np.pi * sigma * sigma * np.log(true_d / true_d[0]) + np.pi * k / 2) / k
        # alpha_k = (np.log(RMAX - k - 2 * sigma * sigma * np.log(true_d / true_d[0])) - np.log(2 * sigma * sigma)) / k
        # alpha_k = np.arctanh((2*sigma*sigma * np.log(true_d/true_d[0]) + k) / RMAX) / k
        # pred_d = parameters.approx_func(d_0, coeffs, False) # пересчитываем приближенные коэффициенты d
        # pred_d = true_d[0] * np.exp(RMAX / (2 * sigma * sigma) * (1 - np.exp(alpha_k * k)))
        pred_d = true_d[0] * np.exp(RMAX / (np.pi * sigma * sigma) * np.tan(alpha_k * k))
        # pred_d = target_approx_func(coeffs[2:], False)
        # pred_d = d_0 * np.exp(alpha * (k ** c))  # пересчитываем приближенные коэффициенты d

        true_d_line.set_data(k, true_d)
        # обновляем координаты линии настоящих коэффициетов d
        true_d_line.set_marker('.')
        true_d_line.set_label('d')
        # обновляем координаты линии приближенных коэффициетов d
        pred_d_line.set_data(k, pred_d)
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
        # aa = sca.val
        # bb = scb.val
        # cc = scc.val
        sigma = coeff_df.index[ssigm.val]
        d_0 = d_df.loc[sigma].iat[0]
        # pred_d = d_0 * np.exp(alpha * (k ** c))
        # pred_d = parameters.approx_func(d_0, [alpha, c, aa, bb, cc], False)
        pred_d = parameters.approx_func(d_0, [alpha, c] + coeff_df.iloc[ssigm.val, 2:-1].tolist(), False)
        # pred_d = parameters.approx_func(d_0, [alpha, c, cc] + coeff_df.iloc[ssigm.val,3:-1].tolist(), False)
        # pred_d = parameters.approx_func(d_0, alpha, c)
        # обновляем координаты линии приближенных коэффициетов d
        # pred_d_line.set_data(k, np.abs(d_list[ssigm.val]) - pred_d )
        pred_d_line.set_data(k, pred_d)
        pred_d_line.set_marker('.')

    ssigm.on_changed(update_sigma)
    # salpha.on_changed(update_params)
    # sc.on_changed(update_params)
    # sca.on_changed(update_params)
    # scb.on_changed(update_params)
    # scc.on_changed(update_params)
    plt.grid()
    plt.show()

def sigma_d_slider2(parameters: Parameters):
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
    ax = plt.axes(xlim=(0, 101), ylim=(-0.2, 2))
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

    sc = Slider(axc, parameters.file_coeffs[1], 0, 50, valstep=0.01)  # ползунок для параметра c
    salpha = Slider(axalpha, parameters.file_coeffs[0], 0, 400, valstep=0.1)  # ползунок для параметра alpha
    ssigm = Slider(axsigm, '$\sigma$', 0, 100, valinit=0, valstep=1)  # ползунок для параметра sigma

    for i in coeff_df.index:  # проходимся по всем сигмам
        d = np.array(d_df.loc[i])
        d_list.append(d)  # записываем в список с коэффициентами d для разных сигм

    def update_sigma(i):
        true_d = np.abs(d_list[i])  # считываем настоящие коэффициенты d
        sigma = coeff_df.index[i]
        coeffs = coeff_df.loc[sigma]

        salpha.set_val(coeffs.iat[0])  # устанавливаем значение параметра alpha
        sc.set_val(coeffs.iat[1])  # устанавливаем значение ползунка с
        # alpha_k = np.arctan(np.pi * sigma * sigma * np.log(true_d / true_d[0]) + np.pi * k / 2) / k
        alpha_pred = ((2 * sigma * sigma * np.log(true_d[-1] / true_d[0])) / RMAX + 1) * (
                1 - (5 * sigma / (2 * sigma * sigma)) ** (-k))
        cot = RMAX / (s:=2 * sigma * sigma) * np.tanh(alpha_pred)

        pred_d = (d_0:=true_d[0]) * np.exp(cot - k/s)
        true_d_line.set_data(k, true_d)
        # обновляем координаты линии настоящих коэффициетов d
        true_d_line.set_marker('.')
        true_d_line.set_label('d')
        # обновляем координаты линии приближенных коэффициетов d
        pred_d_line.set_data(k, pred_d)
        pred_d_line.set_marker('.')
        # xv = np.round(sigma * 6).astype(int)
        # xv = d_0
        # vline.set_data(np.linspace(parameters.bounds[0],parameters.bounds[1],10),np.full(10,xv))
        # print(sigma, xv,np.argmin(true_d[:xv+1]))

        error_text.set_text(('ошибка: ' + str(coeff_df.iat[i, -1])))  # обновляем текст для вывода ошибки
        sigma_text.set_text(('$\sigma$: ' + str(sigmas[i])))  # обновляем текст для вывода значения sigma

        return true_d_line, pred_d_line, error_text, sigma_text,

    ssigm.on_changed(update_sigma)
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
    major_xticks, minor_xticks = np.arange(parameters.bounds[0], parameters.bounds[1] + 1, 1), \
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
def fi_approx_fi_slider2(parameters: Parameters):
    fi_df = pd.read_csv(parameters.parent_directory + '\\' + parameters.phi_vals_filename
                        , header=0
                        , index_col=['sigma']
                        )

    d_df = pd.read_csv(parameters.parent_directory + '\\' + parameters.d_filename
                       , header=0
                       , index_col=['sigma']
                       )

    sigmas = d_df.index.tolist()
    fi = fi_df.loc[0.1]

    x = np.linspace(parameters.bounds[0], parameters.bounds[1], 1000)

    fig = plt.figure()
    ax = plt.axes(xlim=(parameters.bounds[0], parameters.bounds[1]), ylim=(-1.2, 1.2))
    major_xticks, minor_xticks = np.arange(parameters.bounds[0], parameters.bounds[1] + 1, 1), \
                                 np.arange(parameters.bounds[0] - 0.5, parameters.bounds[1] - 0.5, 1)
    major_yticks, minor_yticks = np.arange(-1.2, 1.4, 0.2), np.arange(-1.1, 1.3, 0.2)
    ax.set_xticks(major_xticks)
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(major_yticks)
    ax.set_yticks(minor_yticks, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.8)

    kk = np.arange(0, RMAX + 1)  # значения номеров коэффициентов d соответствующие координате x графика

    d_0 = d_df.iat[0,0]
    sigma = sigmas[0]
    pred_d = (-1) ** kk * d_0 * np.exp(RMAX/(s:=2*sigma*sigma) * (1 - np.exp(-kk/(sigma*sigma))) - kk/s)
    approx_fi = parameters.phi_wave(x, 0.1, d=pred_d)

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
        true_d = np.array(d_df.loc[sigma])
        ind = sindex.val  # считываем значение индекса из значения ползунка

        d_0 = true_d[0]

        s = 2*sigma*sigma
        # alpha_pred = ((2 * sigma * sigma * np.log(true_d[-1] / true_d[0])) / RMAX + 1) * (
        #         1 - (5 * sigma / (2 * sigma * sigma)) ** (-kk))
        # cot = (RMAX /s + np.log(true_d[-1] / true_d[0])) * np.tanh(alpha_pred)
        # pred_d = true_d[0] * np.exp(cot - kk / s)
        pred_d = (-1) ** kk * d_0 * np.exp((np.log(true_d[-1] / true_d[0])+RMAX/s)  * (1 - np.exp(-kk/(sigma*sigma))) - kk/s)

        approx_fi = parameters.phi_wave(x, sigma, d=pred_d)
        # print(approx_fi)
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
    if coeff_num < 0:
        ax = plt.axes(xlim=(0.1, 2.1), ylim=(-3, 1000))
        print(parameters.file_coeffs[0])
        plt.plot('sigma', parameters.file_coeffs[0], '', data=df)
        plt.title(parameters.file_coeffs[0])
    elif coeff_num == 0:
        ax = plt.axes(xlim=(0.1, 2.1), ylim=(0, 5))
        plt.title(parameters.file_coeffs[1])
        plt.plot('sigma', parameters.file_coeffs[1], '', data=df)
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
