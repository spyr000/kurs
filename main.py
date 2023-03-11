import utils
import visual_methods

if __name__ == '__main__':
    p = utils.get_gauss_setup()
    # calculate_alphas2()
    visual_methods.sigma_d_slider(p)
    # calculate_fi()
    visual_methods.fi_approx_fi_slider(p)
    # alpha_sigma_plot()