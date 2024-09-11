import emcee, corner
import numpy as np
from scipy.signal import medfilt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
plt.ion()

import mcmc_functions as mcmc_funcs

def plot_emcee_results():
    # plotting the results of the emcee
    labels = [r"x_b", r"$delta$", r"$alpha1$", r"$alpha2$", r"$const$"]
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

    # changing selected parameters to linear scale
    flat_samples[:,2:4] = np.power(10, flat_samples[:,2:4])

    fig = corner.corner(flat_samples, labels=labels, quantiles=(0.16, 0.5, 0.84), show_titles=True,
                        range=((-1,1),(-5,2),(0,10),(0,25),(-3,3)))
    plt.savefig('mcmc_cornerplot.pdf')

    # converting the delta to log to get the resultant fitted plot
    flat_samples[:,1] = np.power(10, flat_samples[:,1])
    Q=np.quantile(flat_samples,q=[0.5],axis=0).squeeze()

    # plotting the final fitted plot
    plt.figure()
    plt.plot(X, Y, '.k', alpha=0.5, label='data points')
    plt.plot(X, mcmc_funcs.fit_func(X, Q[0], Q[1], Q[2], Q[3], Q[4]), 'r', label='fitted curve')
    plt.axvline(Q[0], color='k', ls='--', label='spectral break')
    plt.grid(True)
    plt.legend()
    plt.xlabel('log10(frequency [Hz])')
    plt.ylabel('log10(spectral power)')

    # plotting some additional curves from the last phase
    samples = sampler.chain[:, -1000:, :].reshape((-1, Nparams))
    samples[:,1:4] = np.power(10, samples[:,1:4])
    for M0, M1, M2, M3, M4 in samples[np.random.randint(len(samples), size=100)]:
        plt.plot(X, mcmc_funcs.fit_func(X, M0, M1, M2, M3, M4), color='b', alpha=0.1)

    plt.savefig('mcmc_spectral_fit.pdf')

    return Q

def scipy_fit(f, x, y):
    popt, pcov = curve_fit(f, x, y, [1.,1.,2, 16.,1.], bounds=([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf]))

    # extracting the fitted parameters from scipy fit (spf)
    xb_spf, delta_spf, alpha1_spf, alpha2_spf, const_spf = popt
    y_spf = f(x, xb_spf, delta_spf, alpha1_spf, alpha2_spf, const_spf)

    plt.figure()
    plt.plot(x, y, '.k', alpha=0.5, label='data points')
    plt.plot(x, y_spf, 'r', label='scipy fit')
    plt.axvline(xb_spf, ls='--', color='k', label='spectral break')
    plt.grid(True)
    plt.legend()
    plt.xlabel('log10(frequency [Hz])')
    plt.ylabel('log10(spectral power)')

    plt.savefig('scipy_spectral_fit.pdf')

    return popt, pcov

def preprocess_data(x, y):
    '''
    Use this function if you want to change which data points you want to use for fitting. 
    e.g.: If you want to set lower and upper limits on the extents in frequency or do an fft filter.
    '''
    # filtering out the nans if any
    nan_mask = np.isnan(y)
    x = x[~nan_mask]
    y = y[~nan_mask]

    y = y[x < np.log10(5)]
    x = x[x < np.log10(5)]
    y = y[x > -1]
    x = x[x > -1]

    # y = y[x < 1]
    # x = x[x < 1]

    return x, y
    

if __name__=='__main__':
    # Xraw, Yraw = np.load('TMP_x1.npy'), np.load('TMP_y1.npy')
    Xraw, Yraw = np.load('DATA_2_FIT.npy').T
    
    # fitting the function
    X = np.log10(Xraw)
    Y = np.log10(Yraw)

    # pre-processing the data
    X, Y = preprocess_data(X, Y)
    Ndata = len(Y)

    #------------------performing scipy fit---------------------------#
    popt_spf, pcov_spf = scipy_fit(mcmc_funcs.fit_func, X, Y)

    #-----------------implementing MCMC fitting-----------------------#
    # limits to initialize walkers | all logs are of base 10
    init_pos = {}
    init_pos['log_xb'] = [-1, 1]
    init_pos['log_delta'] = [-5, 1]
    init_pos['log_alpha1'] = [-2, 2]
    init_pos['log_alpha2'] = [-2, 2]
    init_pos['const'] = [-3, 3]

    Nparams = len(init_pos.keys())
    Nwalkers = 20
    emcee_fitter = mcmc_funcs.mcmc_spectral_fit(init_pos, Nwalkers)

    sampler = emcee.EnsembleSampler(Nwalkers, Nparams, emcee_fitter.log_probability, args=(X, Y))
    sampler.run_mcmc(emcee_fitter.params_init, 20000, progress=True)

    # plotting the mcmc results
    plot_emcee_results()