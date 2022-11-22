import numpy as np
import matplotlib.pyplot as plt
from numba import njit, vectorize, float64, int64

RMAX = 100
BOUND_A, BOUND_B = -np.pi, np.pi


@njit
def D_L(t, sigma):
    return np.sinh(sigma * np.pi) / (sigma * np.pi * np.cosh(sigma * (t - np.pi)))


@njit
def d_L_m(m, sigma):
    N = RMAX
    result = 0
    coeff = 1 / (2 * N + 1)
    for k in range(1, N + 1):
        result += D_L(2 * np.pi * k * coeff, sigma) * np.cos(2 * np.pi * coeff * m * k)
    return coeff * (D_L(0, sigma) + 2 * result)


@njit
def d_list(sigma):
    coeff_list = []
    for m in range(RMAX + 1):
        coeff_list.append(d_L_m(m, sigma))
    return coeff_list


@vectorize([(float64, float64)], target="parallel", nopython=True, cache=True)
def fi_wave(x, sigma):
    d = d_list(sigma)
    result = 0
    for k in range(-RMAX, RMAX + 1):
        result += d[abs(k)] * sigma * sigma / (sigma * sigma + (x - k) * (x - k))
    return result


if __name__ == '__main__':
    sigma = 2
    x = np.linspace(BOUND_A, BOUND_B, RMAX)

    y = fi_wave(x, sigma)
    d = np.abs(d_list(sigma))
    # o = open('coefficients.csv', 'w')
    # for i in range(len(d)):
    #     o.write(f'{i},{d[i]:.3e}\n')
    # o.close()
    fig, ax = plt.subplots(figsize=(12 / 10 * 8, 10 / 12 * 8))
    plt.grid()
    # plt.plot(x, y)
    x_d =  np.arange(len(d))

    plt.plot(x_d,d)
    err_f = []
    a_variations = np.arange(0.4,0.6,0.00000001)
    for i in a_variations:
        error = np.linalg.norm(d[0] * np.exp(-i * np.abs(x_d)) - d)
        err_f.append(error)

    min_num = np.array(err_f).argmin()

    plt.plot(x_d,d[0]*np.exp(-a_variations[min_num]*np.abs(x_d)),label='alpha = '+str(a_variations[min_num]))

    # plt.plot(x_d, d[0] * np.exp(-0.4 * np.abs(x_d)),
    #          label='alpha = 0.4 ' + str(np.linalg.norm(d[0] * np.exp(-0.4 * np.abs(x_d)) - d)))
    # plt.plot(x_d, d[0] * np.exp(-0.5 * np.abs(x_d)),
    #          label='alpha = 0.5 ' + str(np.linalg.norm(d[0] * np.exp(-0.5 * np.abs(x_d)) - d)))
    # plt.plot(x_d, d[0] * np.exp(-0.6 * np.abs(x_d)),
    #          label='alpha = 0.6 ' + str(np.linalg.norm(d[0] * np.exp(-0.6 * np.abs(x_d)) - d)))
    # plt.plot(x_d, d[0] * np.exp(-0.7 * np.abs(x_d)),
    #          label='alpha = 0.7 ' + str(np.linalg.norm(d[0] * np.exp(-0.7 * np.abs(x_d)) - d)))
    # plt.plot(x_d, d[0] * np.exp(-0.8 * np.abs(x_d)),
    #          label='alpha = 0.8' + str(np.linalg.norm(d[0] * np.exp(-0.8 * np.abs(x_d)) - d)))
    plt.legend()
    plt.show()

