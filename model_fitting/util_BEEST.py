'''
This python script is the version following BEEST implementation in terms of likelihood definition. 
'''

import numpy as np
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from scipy.stats import norm
from scipy.special import log_ndtr

def exgaussian_pdf(value, mu, sigma, tau):
    """ 
    ExGaussian log pdf
    """
    if tau > 0.05*sigma:
        z = value - mu - ((sigma**2)/tau)
        return -np.log(tau) - (z+(sigma**2/(2*tau)))/tau + np.log(norm.cdf(z/sigma))
    else: 
        return np.log(norm.pdf(value-mu, scale=sigma))

# change into vectorized 
# point to git repo (our version and our version of their code)

def exgaussian_cdf(value, mu, sigma, tau):
    """
    ExGaussian log cdf upper tail
    """
    if tau > 0.05*sigma:
        z = value - mu - ((sigma**2)/tau)
        return np.log(1-(norm.pdf(value-mu, scale=sigma) - norm.cdf(z/sigma)*np.exp((((mu+((sigma**2)/tau))**2)-
        (mu**2)-2*value *((sigma**2)/tau))/(2*(sigma**2)))))   
    else:
        return np.log((1-(norm.cdf((value-mu)/sigma))))

def eval_exgauss(x, mu_go, sigma_go, tau_go, mu_stop, sigma_stop, tau_stop, p_tf, ssd):
    """
    Integrand function for successful inhibition
    """
    p = (np.exp(exgaussian_cdf(x, mu_go, sigma_go, tau_go)) * 
         np.exp(exgaussian_pdf(x-ssd, mu_stop, sigma_stop, tau_stop)) * 
         (1-p_tf))
    return p

def integrate_exgauss(lower, upper, mu_go, sigma_go, tau_go, 
                     mu_stop, sigma_stop, tau_stop, p_tf, ssd):
    """
    Integrate the integrand function for successful inhibition using SciPy's quad
    (equivalent to gsl_integration_qag)
    """
    # Pack parameters for the integrand
    params = (mu_go, sigma_go, tau_go, mu_stop, sigma_stop, tau_stop, p_tf, ssd)
    
    # Perform the integration
    result, error = integrate.quad(
        eval_exgauss,          # function to integrate
        lower,                 # lower bound
        upper,                 # upper bound
        args=params,           # parameters for the function
        epsabs=1e-4,           # absolute error tolerance
        epsrel=1e-4,           # relative error tolerance
    )
    
    return result

class GoOp(Op):
    """
    PyTensor Op for computing Ex-Gaussian log-likelihood of GoRTs
    """
    __props__ = ()

    def make_node(self, value, mu_go, sigma_go, tau_go):
        # Convert inputs to PyTensor tensors
        value = pt.as_tensor(value, dtype='float64')
        mu_go = pt.as_tensor(mu_go, dtype='float64')
        sigma_go = pt.as_tensor(sigma_go, dtype='float64')
        tau_go = pt.as_tensor(tau_go, dtype='float64')
        
        # Define output as a scalar
        output = [pt.dscalar()]
        return Apply(self, [value, mu_go, sigma_go, tau_go], output)

    def perform(self, node, inputs, outputs):
        value, mu_go, sigma_go, tau_go = inputs

        sum_logp = 0
        for i in range(len(value)):
            p = exgaussian_pdf(value[i], mu_go, sigma_go, tau_go)
            
            if np.isinf(p) or np.isnan(p):
                outputs[0][0] = np.array(-np.inf)
                return
                
            sum_logp += p

        outputs[0][0] = np.array(sum_logp, dtype='float64')

class StopRespondOp(Op):
    """PyTensor Op for computing stop-signal response time log-likelihood."""
    __props__ = ()

    def make_node(self, value, mu_go, sigma_go, tau_go, 
                  mu_stop, sigma_stop, tau_stop, p_tf, ssd):
        # Convert inputs to PyTensor tensors
        value = pt.as_tensor(value, dtype='float64')
        mu_go = pt.as_tensor(mu_go, dtype='float64')
        sigma_go = pt.as_tensor(sigma_go, dtype='float64')
        tau_go = pt.as_tensor(tau_go, dtype='float64')
        mu_stop = pt.as_tensor(mu_stop, dtype='float64')
        sigma_stop = pt.as_tensor(sigma_stop, dtype='float64')
        tau_stop = pt.as_tensor(tau_stop, dtype='float64')
        p_tf = pt.as_tensor(p_tf, dtype='float64')
        ssd = pt.as_tensor(ssd, dtype='float64')

        # Create inputs list
        inputs = [value, mu_go, sigma_go, tau_go, 
                 mu_stop, sigma_stop, tau_stop, p_tf, ssd]

        # Define the output as a scalar
        output = [pt.dscalar()]
        return Apply(self, inputs, output)

    def perform(self, node, inputs, outputs):
        value, mu_go, sigma_go, tau_go, mu_stop, sigma_stop, tau_stop, p_tf, ssd = inputs

        sum_logp = 0
        for i in range(len(value)):
            # Compute the log-likelihood for the failed trigger
            p1 = exgaussian_pdf(value[i], mu_go, sigma_go, tau_go) * p_tf
            # Compute the log-likelihood for the successful trigger
            p2 = exgaussian_pdf(value[i], mu_go, sigma_go, tau_go) * exgaussian_cdf(value[i] - ssd[i], mu_stop, sigma_stop, tau_stop) * (1 - p_tf)
            p = np.log(p1 + p2)
            
            if np.isinf(p) or np.isnan(p):
                outputs[0][0] = np.array(-np.inf)
                return
            
            sum_logp += p
        
        outputs[0][0] = np.array(sum_logp, dtype='float64')

class InhibitionsOp(Op):
    """PyTensor Op for computing censored ExGaussian log-likelihood of inhibitions"""
    __props__ = ()

    def make_node(self, value, mu_go, sigma_go, tau_go, 
                  mu_stop, sigma_stop, tau_stop, p_tf):
        # Convert inputs to PyTensor tensors
        value = pt.as_tensor(value, dtype='int64')  # 2D array of ints
        mu_go = pt.as_tensor(mu_go, dtype='float64')
        sigma_go = pt.as_tensor(sigma_go, dtype='float64')
        tau_go = pt.as_tensor(tau_go, dtype='float64')
        mu_stop = pt.as_tensor(mu_stop, dtype='float64')
        sigma_stop = pt.as_tensor(sigma_stop, dtype='float64')
        tau_stop = pt.as_tensor(tau_stop, dtype='float64')
        p_tf = pt.as_tensor(p_tf, dtype='float64')
        
        inputs = [value, mu_go, sigma_go, tau_go, 
                 mu_stop, sigma_stop, tau_stop, p_tf]
        
        output = [pt.dscalar()]
        return Apply(self, inputs, output)

    def perform(self, node, inputs, outputs):
        value, mu_go, sigma_go, tau_go, mu_stop, sigma_stop, tau_stop, p_tf = inputs
            
        sum_logp = 0
        
        for i in range(value.shape[0]):
            ssd = value[i, 0]
            n_trials = value[i, 1]
            integ_lower = value[i, 2]
            integ_upper = value[i, 3]
                
            # Compute probability for single SSD using integration
            try:
                result = integrate_exgauss(
                    lower=integ_lower,
                    upper=integ_upper,
                    mu_go=mu_go,
                    sigma_go=sigma_go,
                    tau_go=tau_go,
                    mu_stop=mu_stop,
                    sigma_stop=sigma_stop,
                    tau_stop=tau_stop,
                    p_tf=p_tf,
                    ssd=ssd
                )
                
                if result <= 0:
                    outputs[0][0] = np.array(-np.inf)
                    return
                    
                p_ssd = np.log(result)
                
                if np.isinf(p_ssd) or np.isnan(p_ssd):
                    outputs[0][0] = np.array(-np.inf)
                    return
                    
                # Multiply with number of trials and add to overall p
                sum_logp += n_trials * p_ssd
                
            except:
                outputs[0][0] = np.array(-np.inf)
                return
        
        outputs[0][0] = np.array(sum_logp)