from math import exp

import numpy as np
import pandas as pd

import optimization_methods
# from new_gauss import d_list, C
from utils import RMAX
from utils import calc, Parameters, unused


@calc
def calculate_alphas2_and_c(parameters: Parameters, method=optimization_methods.newton, use_csv_vals=True):
    d = pd.read_csv(parameters.parent_directory + '\\' + parameters.d_filename
                    , header=0
                    , index_col=['sigma']
                    )
    coeffs = pd.read_csv(parameters.parent_directory + '\\' + parameters.coeff_filename
                         , header=0
                         , index_col=['sigma']
                         )
    sigmas = np.array(d.index)
    a = []
    alpha, c = 0, 0
    h_a, h_c = 0.05, 0.05
    for sigma in sigmas:
        # if use_csv_vals:
        _alpha, _c, _ = coeffs.loc[sigma]  # начальные значения alpha и c
        # else:
        # _alpha, _c, _ = 0., 0., 0.
        alpha, c, err, h_a, h_c = method(sigma, parameters, alpha=_alpha, c=_c, h_a=h_a,
                                         h_c=h_c)  # получаем новые значения коэффициентов
        while np.isnan(alpha) or np.isnan(c):  # если коэффициенты равны NaN
            print('\n', sigma, '\n')
            ind = np.round((np.round(sigma, 2) - 0.1) / 0.02).astype(int)  # расчитываем индекс сигмы из таблицы
            _alpha, _c, _err = coeffs.iloc[ind - 1]  # получаем значения из предыдущей строки с коэффициентами
            while np.isnan(_alpha):  # если эти значения тоже NaN
                ind -= 1
                _alpha, _c, _err = coeffs.iloc[ind - 1]  # получаем значения из предыдущей строки с коэффициентами
            alpha, c, err, h_a, h_c = method(sigma, parameters, alpha=_alpha, c=_c, h_a=h_a,
                                             h_c=h_c)  # пересчитыеваем значения коэффициентов начиная с другого начального приближения
            # половиним шаги
            h_a /= 2
            h_c /= 2
        # записываем коэффициенты в список
        a.append([sigma, alpha, c, err])
    df = pd.DataFrame(a, columns=['sigma', 'alpha', 'c', 'error'])
    df.set_index('sigma', inplace=True)
    print(df)
    df.to_csv(parameters.parent_directory + '\\' + parameters.coeff_filename)
    return df


@calc
def calculate_d(parameters: Parameters, a=0.1, b=2.1, h=0.02):
    df_list = []
    sigmas = np.arange(a, b + h, h)
    for sigma in sigmas:
        df_list.append(parameters.d_list_func(sigma))
    df = pd.DataFrame(df_list, index=pd.Index(sigmas, name='sigma'))
    df.to_csv(parameters.parent_directory + '\\' + parameters.d_filename)


@unused
@np.vectorize
def approx_fi_wave2(x: np.ndarray, sigma: float, parameters: Parameters):
    kk = np.arange(0, RMAX + 1)
    fi_val = 0
    d_0 = pd.read_csv(parameters.parent_directory + '\\' + parameters.d_filename
                      , header=0
                      , index_col=['sigma']
                      ).loc[sigma].iat[0]
    coeff_df = pd.read_csv(parameters.parent_directory + '\\' + parameters.coeff_filename
                      , header=0
                      , index_col=['sigma']
                      )

    alpha, c, error = coeff_df.loc[sigma]
    if alpha is None or c is None or error is None:
        print('One of the params in None')
        ind = np.round((np.round(sigma, 2) - 0.1) / 0.02).astype(int)
        alpha, c, error = coeff_df.iloc[ind - 1]

    d = d_0 \
        * np.exp(alpha *
                 (kk ** c)) \
        * (-1.) ** kk

    for k in range(-RMAX, RMAX + 1):
        fi_val += d[abs(k)] * exp(-((x - k) * (x - k)) / (2 * sigma * sigma))

    return 1 / C(sigma) * fi_val
