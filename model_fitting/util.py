import pymc.math as pm_math

'''
Likelihood-related functions in pymc
Most of these (pymc) mathematical functions are imported directly from pytensor.tensor.
Doing any kind of math with PyMC random variables, or defining custom likelihoods 
or priors requires using PyTensor expressions rather than NumPy or Python code.
'''
# Define Ex-Gaussian PDF and CDF using PyMC math functions
def exgaussian_pdf_pymc(x, mu, sigma, tau):
    """
    Compute the probability density function (PDF) of the Ex-Gaussian 
    distribution using PyMC math functions.

    Parameters
    ----------
    x : TensorVariable
        The value at which to evaluate the PDF.
    mu : TensorVariable
        The mean of the normal component of the Ex-Gaussian distribution.
    sigma : TensorVariable
        The standard deviation of the normal component of the Ex-Gaussian distribution.
    tau : TensorVariable
        The mean of the exponential component of the Ex-Gaussian distribution.

    Returns
    -------
    TensorVariable
        The value of the PDF at x.
    """
    lambd = 1 / tau
    exp_part = lambd * pm_math.exp((lambd * (mu - x)) + (sigma * lambd)**2 / 2)
    norm_cdf_part = pm_math.erfc((mu - x + sigma**2 * lambd) / (pm_math.sqrt(2) * sigma)) / 2
    return exp_part * norm_cdf_part

def exgaussian_cdf_pymc(x, mu, sigma, tau):
    """
    Compute the cumulative distribution function (CDF) of the
    Ex-Gaussian distribution using PyMC math functions.

    Parameters
    ----------
    x : TensorVariable
        The value at which to evaluate the CDF.
    mu : TensorVariable
        The mean of the normal component of the Ex-Gaussian distribution.
    sigma : TensorVariable
        The standard deviation of the normal component of the Ex-Gaussian distribution.
    tau : TensorVariable
        The mean of the exponential component of the Ex-Gaussian distribution.

    Returns
    -------
    TensorVariable
        The value of the CDF at x.
    """
    lambd = 1 / tau
    norm_cdf_part1 = pm_math.erfc((mu - x) / (pm_math.sqrt(2) * sigma)) / 2
    exp_part = pm_math.exp((sigma * lambd)**2 / 2 - (x - mu) * lambd)
    norm_cdf_part2 = pm_math.erfc((mu - x - sigma**2 * lambd) / (pm_math.sqrt(2) * sigma)) / 2
    return norm_cdf_part1 - exp_part * norm_cdf_part2

# Define the customized likelihood functions using PyMC math functions
def signal_respond_RT_log_likelihood_pymc(t_r, mu_go, sigma_go, tau_go, 
                                          mu_stop, sigma_stop, tau_stop, 
                                          p_tf, ssd_array):
    '''
    Compute the log-likelihood for the 'stop-respond' trials in the stop-signal 
    task using PyMC math functions.

    Parameters
    ----------
    t_r : TensorVariable
        The observed response times for the 'stop-respond' trials.
    mu_go : TensorVariable
        The mean of the normal component for the 'go' trials.
    sigma_go : TensorVariable
        The standard deviation of the normal component for the 'go' trials.
    tau_go : TensorVariable
        The mean of the exponential component for the 'go' trials.
    mu_stop : TensorVariable
        The mean of the normal component for the 'stop' trials.
    sigma_stop : TensorVariable
        The standard deviation of the normal component for the 'stop' trials.
    tau_stop : TensorVariable
        The mean of the exponential component for the 'stop' trials.
    p_tf : TensorVariable
        The probability of a trigger failure.
    ssd_array : TensorVariable
        The stop-signal delays (SSD) for the 'stop-respond' trials.

    Returns
    -------
    TensorVariable
        The total log-likelihood for the 'stop-respond' trials.
    '''
    failed_trigger = p_tf * exgaussian_pdf_pymc(t_r, mu_go, sigma_go, tau_go)
    adjusted_mu_stop = mu_stop + ssd_array
    successful_trigger = (1 - p_tf) * (1 - exgaussian_cdf_pymc(t_r, adjusted_mu_stop, sigma_stop, tau_stop)) * exgaussian_pdf_pymc(t_r, mu_go, sigma_go, tau_go)

    log_likelihoods = pm_math.log(failed_trigger + successful_trigger)
    total_log_likelihood = pm_math.sum(log_likelihoods)
    
    return total_log_likelihood
