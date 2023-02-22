import numpy as np
import pandas as pd

from new_gauss import RMAX
from utils import utility, Container, Parameters


@utility
def is_pos_def(x):
    '''проверка на положительную определенность матрицы 2 x 2'''
    return x[0, 0] > 0 and (x[0, 0] * x[1, 1] - x[0, 1] * x[1, 0]) > 0


def newton(sigma, parameters: Parameters, alpha=0, c=0, lr=1e-8, h_a=0.005, h_c=0.005):
    container = Container()

    d = pd.read_csv(parameters.parent_directory + '\\' + parameters.filename
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
                alpha_derivative = parameters.alpha_derivative(d,d_pred,d_0,alpha,c,k)
                # считаем частную производную ф-ции MSE по переменной c
                c_derivative = parameters.c_derivative(d,d_pred,d_0,alpha,c,k)
                # считаем 2-ю частную производную ф-ции MSE по переменной alpha
                alpha2_derivative = parameters.alpha2_derivative(d, d_pred, d_0, alpha, c, k)
                # считаем 2-ю частную производную ф-ции MSE по переменным alpha и c
                alpha_c_derivative = parameters.alpha_c_derivative(d, d_pred, d_0, alpha, c, k)
                # считаем 2-ю частную производную ф-ции MSE по переменной с
                c2_derivative = parameters.c2_derivative(d, d_pred, d_0, alpha, c, k)
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
