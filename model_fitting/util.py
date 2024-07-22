'''
This python contains likelihood-related functions for pymc model fitting/sampling.
Most of these (pymc) mathematical functions are imported directly from pytensor.tensor.
Doing any kind of math with PyMC random variables, or defining custom likelihoods 
or priors requires using PyTensor expressions rather than NumPy or Python code.
'''
import numpy as np
import pytensor.tensor as pt
import pymc as pm
from pytensor.tensor.extra_ops import unique

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

def precompute_legendre_quadrature(ssd, upper_bound, n):
    """Gauss-Legendre quadrature nodes and weights."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    
    # Transform nodes from [-1, 1] to [lower_bound, upper_bound]
    lower_bound = np.array(ssd)[None, :]  # Expand dims to (1, len(ssd))
    upper_bound = np.array(upper_bound)[:, None]  # Expand dims to (len(upper_bound), 1)
    
    transformed_nodes = 0.5 * (nodes[:, None] + 1) * (upper_bound - lower_bound) + lower_bound
    transformed_weights = weights[:, None] * 0.5 * (upper_bound - lower_bound)

    return transformed_nodes, transformed_weights

def integrate_cexgauss(nodes, weights, mu_go, sigma_go, tau_go, mu_stop, sigma_stop, tau_stop, ssd):
    """
    Numerical integration of the Censored Ex-Gaussian using Gauss-Legendre quadrature.
    """
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
                                      p_tf, ssd, nodes, weights):
    """
    Calculate the log-likelihood for successful inhibition in a stop-signal 
    task using Legendre quadrature rule.

    Parameters
    ----------
    mu_go: float
        Mean of the Gaussian component for the go process.
    sigma_go: float
        Standard deviation of the Gaussian component for the go process.
    tau_go: float
        Mean (or rate parameter) of the exponential component for the go process.
    mu_stop: float
        Mean of the Gaussian component for the stop process.
    sigma_stop: float
        Standard deviation of the Gaussian component for the stop process.
    tau_stop: float
        Mean (or rate parameter) of the exponential component for the stop process.
    p_tf: float
        Probability of triggering the stop process.
    ssd: array-like
        Stop-signal delays.
    nodes: array-like
        Transformed nodes for Legendre quadrature rule calculation.
    weights: array-like
        Transformed weights for Legendre quadrature rule calculation.

    Returns
    -------
    pytensor.tensor
        Log-likelihood of the successful inhibition given the parameters.
    """
    total_integral = integrate_cexgauss(nodes, weights, 
                                        mu_go, sigma_go, tau_go, 
                                        mu_stop, sigma_stop, tau_stop, 
                                        ssd)
    
    # Derive the likelihood for each trial
    likelihood = (1 - p_tf) * total_integral

    # # Get unique SSDs and their counts
    # unique_ssd, unique_indices, inverse_indices, counts = unique(
    #     ssd, return_index=True, return_inverse=True, return_counts=True
    # )

    # # Convert unique_indices and inverse_indices to numpy arrays
    # unique_indices = unique_indices.eval().astype(int)
    # inverse_indices = inverse_indices.eval().astype(int)
    # unique_ssd = unique_ssd.eval()

    # # Compute the integrals for the unique SSDs only once
    # unique_integrals = integrate_cexgauss(nodes[:, unique_indices], weights[:, unique_indices], 
    #                                       mu_go, sigma_go, tau_go, 
    #                                       mu_stop, sigma_stop, tau_stop, 
    #                                       unique_ssd)
    
    # # Sum the integrals for the corresponding indices
    # total_integral = pt.sum(unique_integrals[inverse_indices])
    
    # # Derive the likelihood for each trial
    # likelihood = (1 - p_tf) * total_integral
    
    return pt.log(likelihood)

