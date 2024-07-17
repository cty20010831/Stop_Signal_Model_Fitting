'''
This python contains likelihood-related functions for pymc model fitting/sampling.
Most of these (pymc) mathematical functions are imported directly from pytensor.tensor.
Doing any kind of math with PyMC random variables, or defining custom likelihoods 
or priors requires using PyTensor expressions rather than NumPy or Python code.
'''
import numpy as np
import pytensor.tensor as pt
import pymc as pm

# Sampler: slice?

def stop_respond_likelihood(t_r, mu_go, sigma_go, tau_go, 
                            mu_stop, sigma_stop, tau_stop, 
                            p_tf, ssd):
    """
    Calculate the log-likelihood for a stop-signal response time model 
    using Ex-Gaussian distributions.

    Parameters
    ----------
    t_r : array-like
        Response times.
    mu_go : float
        Mean of the Gaussian component for the go process.
    sigma_go : float
        Standard deviation of the Gaussian component for the go process.
    tau_go : float
        Mean (or rate parameter) of the exponential component for the go process.
    mu_stop : float
        Mean of the Gaussian component for the stop process.
    sigma_stop : float
        Standard deviation of the Gaussian component for the stop process.
    tau_stop : float
        Mean (or rate parameter) of the exponential component for the stop process.
    p_tf : float
        Probability of triggering the stop process.
    ssd : array-like
        Stop-signal delays.
    epsilon : float, optional
        Small constant to avoid numerical instability, by default 1e-10.

    Returns
    -------
    pytensor.tensor
        Log-likelihood of the response times given the parameters.
    """
    # Derive the Ex-Gaussian PDF and CDF using PyMC functions
    exgaussian_go = pm.ExGaussian.dist(mu=mu_go, sigma=sigma_go, nu=tau_go)
    exgaussian_stop = pm.ExGaussian.dist(mu=mu_stop, sigma=sigma_stop, nu=tau_stop)
    
    failed_trigger = p_tf * pt.exp(pm.logp(exgaussian_go, t_r))
    
    # Focus on the relevant time interval for determining if the stop process 
    # finished before the response was made (specifically, t_r - ssd_array)
    successful_trigger = (1 - p_tf) * (1 - pt.exp(pm.logcdf(exgaussian_stop, t_r - ssd))) * pt.exp(pm.logp(exgaussian_go, t_r))
    
    likelihood = failed_trigger + successful_trigger
    
    return pt.log(likelihood)

def legendre_quadrature_pytensor(ssd, upper_bound, n):
    """Gauss-Legendre quadrature nodes and weights."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    nodes_pt, weights_pt = pt.constant(nodes), pt.constant(weights)
    
    # Transform nodes from [-1, 1] to [lower_bound, upper_bound]
    # lower_bound = pt.as_tensor_variable(lower_bound).dimshuffle('x')  # Expand dims to (1,)
    lower_bound = ssd
    upper_bound = pt.as_tensor_variable(upper_bound).dimshuffle('x')  # Expand dims to (1,)
    
    transformed_nodes = 0.5 * (nodes_pt[:, None] + 1) * (upper_bound - lower_bound) + lower_bound
    transformed_weights = weights_pt[:, None] * 0.5 * (upper_bound - lower_bound)

    return transformed_nodes, transformed_weights

def integrate_cexgauss(upper_bound, mu_go, sigma_go, tau_go, mu_stop, 
                       sigma_stop, tau_stop, ssd, n):
    """Numerical integration of the Censored Ex-Gaussian using Gauss-Legendre quadrature."""
    nodes, weights = legendre_quadrature_pytensor(ssd, upper_bound, n)

    # Derive the Ex-Gaussian PDF and CDF using PyMC functions
    exgaussian_go = pm.ExGaussian.dist(mu=mu_go, sigma=sigma_go, nu=tau_go)
    exgaussian_stop = pm.ExGaussian.dist(mu=mu_stop, sigma=sigma_stop, nu=tau_stop)

    exgaussian_cdf_vals = pt.exp(pm.logcdf(exgaussian_go, nodes))
    exgaussian_pdf_vals = pt.exp(pm.logp(exgaussian_stop, nodes - ssd))

    integrands = (1 - exgaussian_cdf_vals) * exgaussian_pdf_vals
    integrals = weights * integrands
    total_integral = pt.sum(integrals, axis=0)

    return total_integral

def successful_inhibit_log_likelihood(mu_go, sigma_go, tau_go, 
                                      mu_stop, sigma_stop, tau_stop, 
                                      p_tf, ssd, upper_bound=10000, n=100):
    """
    Calculate the log-likelihood for successful inhibition in a stop-signal 
    task using Legendre quadrature rule.

    Parameters
    ----------
    mu_go : float
        Mean of the Gaussian component for the go process.
    sigma_go : float
        Standard deviation of the Gaussian component for the go process.
    tau_go : float
        Mean (or rate parameter) of the exponential component for the go process.
    mu_stop : float
        Mean of the Gaussian component for the stop process.
    sigma_stop : float
        Standard deviation of the Gaussian component for the stop process.
    tau_stop : float
        Mean (or rate parameter) of the exponential component for the stop process.
    p_tf : float
        Probability of triggering the stop process.
    upper_bound : array-like
        Upper bound of the integration interval.
    ssd : array-like
        Stop-signal delays.
    n : int, optional
        Number of quadrature nodes, by default 50.
    epsilon : float, optional
        Small constant to avoid numerical instability, by default 1e-10.

    Returns
    -------
    pytensor.tensor
        Log-likelihood of the successful inhibition given the parameters.
    """
    total_integral = integrate_cexgauss(upper_bound, mu_go, sigma_go, tau_go, mu_stop, sigma_stop, tau_stop, ssd, n)

    # Derive the likelihood for each trial
    likelihood = (1 - p_tf) * total_integral
    
    return pt.log(likelihood)