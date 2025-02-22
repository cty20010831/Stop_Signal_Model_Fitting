'''
This python script is the outdated version of defining the two custom likelihood 
functions for stop respond and successful inhibition trials. Specifically, it 
uses pytensor Op class to define likelihood functions by integrating numpy/scipy 
functions (e.g., `exgaussian_pdf`, `exgaussian_cdf`).

The main reason why this script is outdated is that the default 
`Distribution.logp` and `Distribution.logcdf` of Ex-Gaussian Distribution in pymc
helps avoid the overflow issues encountered exponential operations in the pdf 
and cdf Ex-Gaussian distribution written in numpy.
'''

import numpy as np
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from scipy.stats import norm

# Define Ex-Gaussian PDF and CDF using numpy functions for the perform method
def exgaussian_pdf(x, mu, sigma, tau):
    """
    Calculate the probability density function (PDF) 
    of the Ex-Gaussian distribution.

    Parameters
    ----------
    x : array-like
        Values at which to evaluate the PDF.
    mu : float
        Mean of the Gaussian component.
    sigma : float
        Standard deviation of the Gaussian component.
    tau : float
        Mean (or rate parameter) of the exponential component.

    Returns
    -------
    array-like
        PDF values of the Ex-Gaussian distribution at `x`.
    """
    lambd = 1 / tau
    exp_part = lambd * np.exp((lambd * (mu - x)) + (sigma * lambd)**2 / 2)
    norm_cdf_part = norm.cdf((x - mu) / sigma - lambd * sigma)
    return exp_part * norm_cdf_part

def exgaussian_cdf(x, mu, sigma, tau):
    """
    Calculate the cumulative distribution function (CDF) 
    of the Ex-Gaussian distribution.

    Parameters
    ----------
    x : array-like
        Values at which to evaluate the CDF.
    mu : float
        Mean of the Gaussian component.
    sigma : float
        Standard deviation of the Gaussian component.
    tau : float
        Mean (or rate parameter) of the exponential component.

    Returns
    -------
    array-like
        CDF values of the Ex-Gaussian distribution at `x`.
    """
    lambd = 1 / tau
    norm_cdf_part1 = norm.cdf((x - mu) / sigma)
    exp_part = np.exp((sigma * lambd)**2 / 2 - (x - mu) * lambd)
    norm_cdf_part2 = norm.cdf((x - mu) / sigma - sigma * lambd)
    return norm_cdf_part1 - exp_part * norm_cdf_part2

def stop_respond_likelihood(t_r, mu_go, sigma_go, tau_go, 
                            mu_stop, sigma_stop, tau_stop, 
                            p_tf, ssd, epsilon=1e-10):
    """
    Calculate the log-likelihood for a stop-signal response time 
    model using Ex-Gaussian distributions.

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
    np.ndarray
        Log-likelihood of the response times given the parameters.
    """
    failed_trigger = p_tf * exgaussian_pdf(t_r, mu_go, sigma_go, tau_go)
    # Focus on the relevant time interval for determining if the stop process 
    # finished before the response was made (specifically, t_r - ssd_array)
    successful_trigger = (1 - p_tf) * (1 - exgaussian_cdf(t_r - ssd, mu_stop, sigma_stop, tau_stop)) * exgaussian_pdf(t_r, mu_go, sigma_go, tau_go)
    
    likelihood = failed_trigger + successful_trigger
    
    return np.log(likelihood + epsilon)

class Stop_Respond(Op):
    __props__ = ()

    def make_node(self, it_r, imu_go, isigma_go, itau_go, 
                  imu_stop, isigma_stop, itau_stop, ip_tf, issd) -> Apply:
        """Create a computation graph node for the stop-signal response time model.

        Parameters
        ----------
        it_r : array-like
            Response times.
        imu_go : float
            Mean of the Gaussian component for the go process.
        isigma_go : float
            Standard deviation of the Gaussian component for the go process.
        itau_go : float
            Mean (or rate parameter) of the exponential component for the go process.
        imu_stop : float
            Mean of the Gaussian component for the stop process.
        isigma_stop : float
            Standard deviation of the Gaussian component for the stop process.
        itau_stop : float
            Mean (or rate parameter) of the exponential component for the stop process.
        ip_tf : float
            Probability of triggering the stop process.
        issd : array-like
            Stop-signal delays.

        Returns
        -------
        Apply
            A PyTensor Apply node representing the stop-signal response time model.
        """
        it_r = pt.as_tensor(it_r, dtype='float64')
        imu_go = pt.as_tensor(imu_go, dtype='float64')
        isigma_go = pt.as_tensor(isigma_go, dtype='float64')
        itau_go = pt.as_tensor(itau_go, dtype='float64')
        imu_stop = pt.as_tensor(imu_stop, dtype='float64')
        isigma_stop = pt.as_tensor(isigma_stop, dtype='float64')
        itau_stop = pt.as_tensor(itau_stop, dtype='float64')
        ip_tf = pt.as_tensor(ip_tf, dtype='float64')
        issd = pt.as_tensor(issd, dtype='float64')
        
        inputs = [it_r, imu_go, isigma_go, itau_go, 
                  imu_stop, isigma_stop, itau_stop, ip_tf, issd]
        
        # Output the log likelihood (for each trial)
        output = [issd.type()]

        return Apply(self, inputs, output)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]):
        """Compute the log-likelihood of the stop-signal response time model.

        Parameters
        ----------
        node : Apply
            The computation graph node.
        inputs : list of np.ndarray
            Input values for the node.
        outputs : list of list of None
            Outputs for the node.

        Returns
        -------
        None
        """
        # Access input nodes
        it_r, imu_go, isigma_go, itau_go, imu_stop, isigma_stop, itau_stop, ip_tf, issd = inputs

        # Calculate sucessful inhition likelihood
        result = stop_respond_likelihood(
            t_r=it_r, mu_go=imu_go, sigma_go=isigma_go, tau_go=itau_go, 
            mu_stop=imu_stop, sigma_stop=isigma_stop, tau_stop=itau_stop, 
            p_tf=ip_tf, ssd=issd
        )

        outputs[0][0] = np.array(result)

def legendre_quadrature(upper_bound, ssd, n=50):
    """Perform Gauss-Legendre quadrature to approximate the integral for given bounds.

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
    
    # Transform nodes from [-1, 1] (default for the legendre quadrature rule) 
    # to [SSD, upper_bound]
    # Reshape the node array into (n, 1) and ensure upper_bound is a scalar or has the same shape as ssd
    upper_bound = upper_bound[np.newaxis]
    
    transformed_nodes = 0.5 * (nodes[:, np.newaxis] + 1) * (upper_bound - ssd) + ssd
    transformed_weights = weights[:, np.newaxis] * 0.5 * (upper_bound - ssd)

    return transformed_nodes, transformed_weights

def successful_inhibit_log_likelihood(mu_go, sigma_go, tau_go, 
                                      mu_stop, sigma_stop, tau_stop, 
                                      p_tf, upper_bound, ssd, n=50, epsilon=1e-10):
    """
    Calculate the log-likelihood for successful inhibition in a 
    stop-signal task using Legendre quadrature.

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
    np.ndarray
        Log-likelihood of the successful inhibition given the parameters.
    """
    total_integral = 0
    nodes, weights = legendre_quadrature(upper_bound, ssd, n)

    for i in range(n):
        exgaussian_cdf_val = exgaussian_cdf(nodes[i], mu_go, sigma_go, tau_go)
        exgaussian_pdf_val = exgaussian_pdf(nodes[i] - ssd, mu_stop, sigma_stop, tau_stop)
        integrand = (1 - exgaussian_cdf_val) * exgaussian_pdf_val
        integral = weights[i] * integrand
        total_integral += integral

    likelihood = (1 - p_tf) * total_integral

    return np.log(likelihood + epsilon)

class Succcessful_Inhibition(Op):
    __props__ = ()

    def make_node(self, iupper_bound, imu_go, isigma_go, itau_go, 
                  imu_stop, isigma_stop, itau_stop, ip_tf, issd) -> Apply:
        """Create a computation graph node for the successful inhibition model.

        Parameters
        ----------
        iupper_bound : array-like
            Upper bound of the integration interval.
        imu_go : float
            Mean of the Gaussian component for the go process.
        isigma_go : float
            Standard deviation of the Gaussian component for the go process.
        itau_go : float
            Mean (or rate parameter) of the exponential component for the go process.
        imu_stop : float
            Mean of the Gaussian component for the stop process.
        isigma_stop : float
            Standard deviation of the Gaussian component for the stop process.
        itau_stop : float
            Mean (or rate parameter) of the exponential component for the stop process.
        ip_tf : float
            Probability of triggering the stop process.
        issd : array-like
            Stop-signal delays.

        Returns
        -------
        Apply
            A PyTensor Apply node representing the successful inhibition model.
        """
        iupper_bound = pt.as_tensor(iupper_bound, dtype='float64')
        imu_go = pt.as_tensor(imu_go, dtype='float64')
        isigma_go = pt.as_tensor(isigma_go, dtype='float64')
        itau_go = pt.as_tensor(itau_go, dtype='float64')
        imu_stop = pt.as_tensor(imu_stop, dtype='float64')
        isigma_stop = pt.as_tensor(isigma_stop, dtype='float64')
        itau_stop = pt.as_tensor(itau_stop, dtype='float64')
        ip_tf = pt.as_tensor(ip_tf, dtype='float64')
        issd = pt.as_tensor(issd, dtype='float64')
        
        inputs = [iupper_bound, imu_go, isigma_go, itau_go, 
                  imu_stop, isigma_stop, itau_stop, ip_tf, issd]
        
        # Output as the total log likelihood (for each participant)
        output = [issd.type()]

        return Apply(self, inputs, output)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]):
        """Compute the log-likelihood of the successful inhibition model.

        Parameters
        ----------
        node : Apply
            The computation graph node.
        inputs : list of np.ndarray
            Input values for the node.
        outputs : list of list of None
            Outputs for the node.

        Returns
        -------
        None
        """
        # Access input nodes
        iupper_bound, imu_go, isigma_go, itau_go, imu_stop, isigma_stop, itau_stop, ip_tf, issd = inputs

        # Calculate sucessful inhition likelihood
        result = successful_inhibit_log_likelihood(
            mu_go=imu_go, sigma_go=isigma_go, tau_go=itau_go, 
            mu_stop=imu_stop, sigma_stop=isigma_stop, tau_stop=itau_stop, 
            p_tf=ip_tf, upper_bound=iupper_bound, ssd=issd
        )

        outputs[0][0] = np.array(result)

