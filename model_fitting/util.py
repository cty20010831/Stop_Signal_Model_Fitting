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
                            p_tf, ssd, epsilon=1e-10):
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
    
    return pt.log(likelihood + epsilon)

def legendre_quadrature_pytensor(upper_bound, ssd, n=50):
    """
    Perform Gauss-Legendre quadrature to approximate the 
    definite integral of a function.

    Parameters
    ----------
    upper_bound : array-like
        Upper bound of the integration interval.
    ssd : array-like
        Stop-signal delays.
    n : int, optional
        Number of quadrature nodes, by default 50.

    Returns
    -------
    tuple
        Transformed nodes and weights for the quadrature.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)
    nodes_pt, weights_pt = pt.constant(nodes), pt.constant(weights)
    
    # Transform nodes from [-1, 1] (default for the legendre quadrature rule) 
    # to [SSD, upper_bound]
    transformed_nodes = 0.5 * (nodes_pt[:, None] + 1) * (upper_bound - ssd) + ssd
    transformed_weights = weights_pt[:, None] * 0.5 * (upper_bound - ssd)

    return transformed_nodes, transformed_weights

def successful_inhibit_log_likelihood(mu_go, sigma_go, tau_go, 
                                      mu_stop, sigma_stop, tau_stop, 
                                      p_tf, upper_bound, ssd, n=50, epsilon=1e-10):
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
    # Derive the nodes and weights for Gauss-Legendre quadrature
    nodes, weights = legendre_quadrature_pytensor(upper_bound, ssd, n)

    # Derive the Ex-Gaussian PDF and CDF using PyMC functions
    exgaussian_go = pm.ExGaussian.dist(mu=mu_go, sigma=sigma_go, nu=tau_go)
    exgaussian_stop = pm.ExGaussian.dist(mu=mu_stop, sigma=sigma_stop, nu=tau_stop)

    exgaussian_cdf_vals = pt.exp(pm.logcdf(exgaussian_go, nodes))
    exgaussian_pdf_vals = pt.exp(pm.logp(exgaussian_stop, nodes - ssd))

    # Sum (integration) of the product of transformed nodes and weights
    integrands = (1 - exgaussian_cdf_vals) * exgaussian_pdf_vals
    integrals = weights * integrands
    total_integral = pt.sum(integrals, axis=0)

    # Derive the likelihood for each trial
    likelihood = (1 - p_tf) * total_integral

    return pt.log(likelihood + epsilon)