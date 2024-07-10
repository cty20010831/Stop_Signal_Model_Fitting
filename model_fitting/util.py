import numpy as np
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from scipy.stats import norm

'''
Likelihood-related functions in pymc
Most of these (pymc) mathematical functions are imported directly from pytensor.tensor.
Doing any kind of math with PyMC random variables, or defining custom likelihoods 
or priors requires using PyTensor expressions rather than NumPy or Python code.
'''
# Define Ex-Gaussian PDF and CDF using numpy functions for the perform method
def exgaussian_pdf_numpy(x, mu, sigma, tau):
    lambd = 1 / tau
    exp_part = lambd * np.exp((lambd * (mu - x)) + (sigma * lambd)**2 / 2)
    norm_cdf_part = norm.cdf((x - mu) / sigma - lambd * sigma)
    return exp_part * norm_cdf_part

def exgaussian_cdf_numpy(x, mu, sigma, tau):
    lambd = 1 / tau
    norm_cdf_part1 = norm.cdf((x - mu) / sigma)
    exp_part = np.exp((sigma * lambd)**2 / 2 - (x - mu) * lambd)
    norm_cdf_part2 = norm.cdf((x - mu) / sigma - sigma * lambd)
    return norm_cdf_part1 - exp_part * norm_cdf_part2

# def stop_respond_likelihood(t_r, mu_go, sigma_go, tau_go, 
#                             mu_stop, sigma_stop, tau_stop, 
#                             p_tf, ssd_array):
    
#     failed_trigger = p_tf * exgaussian_pdf_numpy(t_r, mu_go, sigma_go, tau_go)
#     # Focus on the relevant time interval for determining if the stop process 
#     # finished before the response was made (specifically, t_r - ssd_array)
#     successful_trigger = (1 - p_tf) * (1 - exgaussian_cdf_numpy(t_r - ssd_array, mu_stop, sigma_stop, tau_stop)) * exgaussian_pdf_numpy(t_r, mu_go, sigma_go, tau_go)

#     log_likelihoods = np.log(failed_trigger + successful_trigger)
#     total_log_likelihood = np.sum(log_likelihoods)
    
#     return total_log_likelihood

def stop_respond_likelihood(t_r, mu_go, sigma_go, tau_go, 
                            mu_stop, sigma_stop, tau_stop, 
                            p_tf, ssd):
    
    failed_trigger = p_tf * exgaussian_pdf_numpy(t_r, mu_go, sigma_go, tau_go)
    # Focus on the relevant time interval for determining if the stop process 
    # finished before the response was made (specifically, t_r - ssd_array)
    successful_trigger = (1 - p_tf) * (1 - exgaussian_cdf_numpy(t_r - ssd, mu_stop, sigma_stop, tau_stop)) * exgaussian_pdf_numpy(t_r, mu_go, sigma_go, tau_go)
    
    likelihood = failed_trigger + successful_trigger
    
    return np.log(likelihood)

# class Stop_Respond(Op):
#     __props__ = ()

#     def make_node(self, it_r, imu_go, isigma_go, itau_go, 
#                   imu_stop, isigma_stop, itau_stop, ip_tf, issd_array) -> Apply:
#         it_r = pt.as_tensor(it_r, dtype='float64')
#         imu_go = pt.as_tensor(imu_go, dtype='float64')
#         isigma_go = pt.as_tensor(isigma_go, dtype='float64')
#         itau_go = pt.as_tensor(itau_go, dtype='float64')
#         imu_stop = pt.as_tensor(imu_stop, dtype='float64')
#         isigma_stop = pt.as_tensor(isigma_stop, dtype='float64')
#         itau_stop = pt.as_tensor(itau_stop, dtype='float64')
#         ip_tf = pt.as_tensor(ip_tf, dtype='float64')
#         issd_array = pt.as_tensor(issd_array, dtype='float64')
        
#         inputs = [it_r, imu_go, isigma_go, itau_go, 
#                   imu_stop, isigma_stop, itau_stop, ip_tf, issd_array]
        
#         # Output as the total log likelihood (for each participant)
#         output = [pt.dscalar()]

#         return Apply(self, inputs, output)

#     def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]):
#         # Access input nodes
#         it_r, imu_go, isigma_go, itau_go, imu_stop, isigma_stop, itau_stop, ip_tf, issd_array = inputs

#         # Calculate sucessful inhition likelihood
#         result = stop_respond_likelihood(
#             t_r=it_r, mu_go=imu_go, sigma_go=isigma_go, tau_go=itau_go, 
#             mu_stop=imu_stop, sigma_stop=isigma_stop, tau_stop=itau_stop, 
#             p_tf=ip_tf, ssd_array=issd_array
#         )

#         outputs[0][0] = np.array(result)

class Stop_Respond(Op):
    __props__ = ()

    def make_node(self, it_r, imu_go, isigma_go, itau_go, 
                  imu_stop, isigma_stop, itau_stop, ip_tf, issd) -> Apply:
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
    nodes, weights = np.polynomial.legendre.leggauss(n)
    
    # Transform nodes from [-1, 1] (default for the legendre quadrature rule) 
    # to [SSD, upper_bound]
    # Reshape the node array into (n, 1) and ensure upper_bound is a scalar or has the same shape as ssd
    upper_bound = upper_bound[np.newaxis]
    
    transformed_nodes = 0.5 * (nodes[:, np.newaxis] + 1) * (upper_bound - ssd) + ssd
    transformed_weights = weights[:, np.newaxis] * 0.5 * (upper_bound - ssd)

    return transformed_nodes, transformed_weights

# def successful_inhibit_log_likelihood(mu_go, sigma_go, tau_go, 
#                                       mu_stop, sigma_stop, tau_stop, 
#                                       p_tf, upper_bound, ssd_array, n=50):
#     individual_likelihoods = []

#     for ssd in ssd_array:
#         total_integral = 0
#         nodes, weights = legendre_quadrature(upper_bound, ssd, n)

#         for j in range(n):
#             integrand = (1 - exgaussian_cdf_numpy(nodes[j], mu_go, sigma_go, tau_go)) * exgaussian_pdf_numpy(nodes[j] - ssd, mu_stop, sigma_stop, tau_stop)
#             integral = weights[j] * integrand
#             total_integral += integral

#         individual_likelihood = (1 - p_tf) * total_integral
#         individual_likelihoods.append(individual_likelihood)

#     log_likelihoods = np.log(individual_likelihoods)
#     total_log_likelihood = np.sum(log_likelihoods)

#     return total_log_likelihood

def successful_inhibit_log_likelihood(mu_go, sigma_go, tau_go, 
                                      mu_stop, sigma_stop, tau_stop, 
                                      p_tf, upper_bound, ssd, n=50, epsilon=1e-10):

    total_integral = 0
    nodes, weights = legendre_quadrature(upper_bound, ssd, n)

    for i in range(n):
        exgaussian_cdf_val = exgaussian_cdf_numpy(nodes[i], mu_go, sigma_go, tau_go)
        exgaussian_pdf_val = exgaussian_pdf_numpy(nodes[i] - ssd, mu_stop, sigma_stop, tau_stop)
        integrand = (1 - exgaussian_cdf_val) * exgaussian_pdf_val
        integral = weights[i] * integrand
        total_integral += integral

    likelihood = (1 - p_tf) * total_integral

    return np.log(likelihood + epsilon)
        

# class Succcessful_Inhibition(Op):
#     __props__ = ()

#     def make_node(self, iupper_bound, imu_go, isigma_go, itau_go, 
#                   imu_stop, isigma_stop, itau_stop, ip_tf, issd_array) -> Apply:
#         iupper_bound = pt.as_tensor(iupper_bound, dtype='float64')
#         imu_go = pt.as_tensor(imu_go, dtype='float64')
#         isigma_go = pt.as_tensor(isigma_go, dtype='float64')
#         itau_go = pt.as_tensor(itau_go, dtype='float64')
#         imu_stop = pt.as_tensor(imu_stop, dtype='float64')
#         isigma_stop = pt.as_tensor(isigma_stop, dtype='float64')
#         itau_stop = pt.as_tensor(itau_stop, dtype='float64')
#         ip_tf = pt.as_tensor(ip_tf, dtype='float64')
#         issd_array = pt.as_tensor(issd_array, dtype='float64')
        
#         inputs = [iupper_bound, imu_go, isigma_go, itau_go, 
#                   imu_stop, isigma_stop, itau_stop, ip_tf, issd_array]
        
#         # Output as the total log likelihood (for each participant)
#         output = [pt.dscalar()]

#         return Apply(self, inputs, output)

#     def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]):
#         # Access input nodes
#         iupper_bound, imu_go, isigma_go, itau_go, imu_stop, isigma_stop, itau_stop, ip_tf, issd_array = inputs

#         # Calculate sucessful inhition likelihood
#         result = successful_inhibit_log_likelihood(
#             mu_go=imu_go, sigma_go=isigma_go, tau_go=itau_go, 
#             mu_stop=imu_stop, sigma_stop=isigma_stop, tau_stop=itau_stop, 
#             p_tf=ip_tf, upper_bound=iupper_bound, ssd_array=issd_array
#         )

#         outputs[0][0] = np.array(result)

class Succcessful_Inhibition(Op):
    __props__ = ()

    def make_node(self, iupper_bound, imu_go, isigma_go, itau_go, 
                  imu_stop, isigma_stop, itau_stop, ip_tf, issd) -> Apply:
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
        # Access input nodes
        iupper_bound, imu_go, isigma_go, itau_go, imu_stop, isigma_stop, itau_stop, ip_tf, issd = inputs

        # Calculate sucessful inhition likelihood
        result = successful_inhibit_log_likelihood(
            mu_go=imu_go, sigma_go=isigma_go, tau_go=itau_go, 
            mu_stop=imu_stop, sigma_stop=isigma_stop, tau_stop=itau_stop, 
            p_tf=ip_tf, upper_bound=iupper_bound, ssd=issd
        )

        outputs[0][0] = np.array(result)