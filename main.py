import os

import calc_methods
import utils
import visual_methods
from gauss import loss_funcs

if __name__ == '__main__':
    p = utils.get_gauss_setup()
    p.coeff_filename='test_gauss.csv'
    loss_funcs.calculate_new_vals(p)
    visual_methods.sigma_d_slider(p)
    # calculate_fi()
    # calc_methods.calculate_fi(p)
    visual_methods.fi_approx_fi_slider(p)
    visual_methods.alpha_sigma_plot(p,2)