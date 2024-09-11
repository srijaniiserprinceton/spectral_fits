import numpy as np

class mcmc_spectral_fit:
    def __init__(self, init_pos, Nwalkers=20):
        self.init_pos = init_pos
        self.Nwalkers = Nwalkers
        self.Nparams = len(self.init_pos.keys())
        self.params_init = self.initialize_walkers()

    def log_prior(self, model_params):
        models_inrange = True
        
        for param_idx, param_key in enumerate(self.init_pos.keys()):
            models_inrange *= self.init_pos[param_key][0] < model_params[param_idx] < self.init_pos[param_key][1]

        if models_inrange: 
            return 0.0
        return -np.inf

    def log_probability(self, model_params, xdata, ydata):
        lp = self.log_prior(model_params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(model_params, xdata, ydata)

    def log_likelihood(self, model_params, xdata, ydata):
        # customize this if you have different parameterization
        log_xb, log_delta, log_alpha1, log_alpha2, const = model_params
        delta = np.power(10, log_delta)
        alpha1  = np.power(10, log_alpha1)
        alpha2 = np.power(10, log_alpha2)

        # building the model prediction
        model = fit_func(xdata, log_xb, delta, alpha1, alpha2, const)

        # the cost function
        cost_mismatch = np.sum((ydata - model)**4)
        return -0.5 * cost_mismatch

    def initialize_walkers(self):
        params_init = np.zeros((self.Nwalkers, self.Nparams))

        for param_idx, param_key in enumerate(self.init_pos.keys()):
            param_min, param_max = self.init_pos[param_key]
            params_init[:,param_idx] = np.random.rand(self.Nwalkers) * (param_max - param_min) + param_min
        
        return params_init


def fit_func(xdata, xb, delta, alpha1, alpha2, const):
    # changing this function if you define a different functional form
    return const - alpha1 * (xdata - xb) + (alpha1 - alpha2) * delta * np.log10(0.5 * (1 + np.power(10, (xdata - xb)/delta)))