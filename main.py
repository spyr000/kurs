import os

import calc_methods
import utils
import visual_methods
from gauss import loss_funcs

if __name__ == '__main__':
    p = utils.get_lorenz_setup()
    p.coeff_filename='test_lorenz.csv'
    # visual_methods.calculate_hyperbola(p)
    visual_methods.check_ratio(p)
    # loss_funcs.calculate_new_vals(p)
    # # p.approx_func = lambda d_0, x, s=True: utils.gauss_approx_func(d_0, x, s) + utils.target_approx_func(x[2:], s)
    # visual_methods.sigma_d_slider2(p)
    # # calculate_fi()
    # # calc_methods.calculate_fi(p)
    visual_methods.fi_approx_fi_slider2(p)
    # # visual_methods.alpha_sigma_plot(p,0)