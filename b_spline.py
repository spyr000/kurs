import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: []})


def B(x, n):
    if n == 2:
        if (x >= -1 and x <= 1):
            return 1 - abs(x)
        else:
            return 0
    result = ((n + 2 * x) / (2 * (n - 1))) * B(x + 0.5, n - 1) + ((n - 2 * x) / (2 * (n - 1))) * B(x - 0.5, n - 1)
    return result


def phi(t, n):
    sum = B(0, n)
    for k in range(1, math.floor(n / 2) + 1):
        sum += 2 * B(k, n) * np.cos(k * t)
    return sum


def d(phi):
    return 1 / phi


def d_b(m, D_k, N):
    d_b_m = D_k[0]
    for k in range(1, N + 1):
        d_b_m += 2 * D_k[k] * np.cos(2 * np.pi * k * m / (2 * N + 1))
    return d_b_m / (2 * N + 1)


def fi_wav(x, d_b_m, n):
    fi = 0
    for k in range(-N, N + 1):
        fi += d_b_m[abs(k)] * B(x - k, n)
    return fi


def write_d_to_csv():
    N = 100
    num = 12
    x = np.arange(3, num + 1)
    df_list = []
    k = np.arange(N + 1)
    for n in x:
        D_k = []

        for k_ in k:
            D_k.append(1 / (phi(2 * np.pi * k_ / (2 * N + 1), n)))
        D_k = np.array(D_k)
        d_b_m = []
        for m in k:
            d_b_m.append(d_b(m, D_k, N))
        d_b_m = np.abs(d_b_m)
        df_list.append(d_b_m)
    df = pd.DataFrame(df_list, index=pd.Index(x, name='n'))
    df.to_csv('b_splines_coeffs.csv')
    print(df)


def write_coeffs_to_csv():
    d_df = pd.read_csv('b_splines_coeffs.csv', header=0, index_col=['n'])
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
    df.to_csv('n-alpha-norm_0-2.csv', index=False)


if __name__ == '__main__':
    write_coeffs_to_csv()
