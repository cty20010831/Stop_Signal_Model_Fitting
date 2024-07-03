import numpy as np
import pandas as pd
import random
from scipy.stats import norm, exponnorm, truncnorm

'''
Functions related to simulation
'''
def simulate_exgaussian(mu, sigma, tau):
    '''
    Generate a random value following Ex-Gaussian distribution with given parameters

    Inputs:
        1) mu: mean of the Gaussian component
        2) sigma: standard deviation of the Gaussian component
        3) tau: mean (or rate parameter) of the exponential component
    '''
    
    # Kappa is the shape parameter for the exponential part
    kappa = tau / sigma
    # Simulate Ex-Gaussian distribution using scipy's exponnorm which is a combination of exponential and normal distributions
    simulated_value = exponnorm.rvs(K=kappa, loc=mu, scale=sigma, size=1)[0]
    
    return simulated_value

def random_normal_truncate(mu, sigma, lower_bound=-np.inf, upper_bound=np.inf):
    '''
    Generates a random value for a truncated normal distribution.

    Inputs:
        1) mu: mean of the original normal distribution
        2) sigma: sd of the original normal distribution
        3) lower_bound: lower bound for the returned value
        4) upper_bound: higher bound for the returned value

    Return: 
        Simulated value
    '''
    a, b = (lower_bound - mu) / sigma, (upper_bound - mu) / sigma
    return truncnorm(a, b, loc=mu, scale=sigma).rvs()

def simulate_p_tf_hierarchical(N):
    """
    Simulates the participant-level P(TF) parameters (in hierarchical case) for N participants.

    Args:
        N (int): Number of participants

    Returns:
        np.array: Array of adjusted probit-transformed P(TF) values for all participants.
    """
    # Generate initial probabilities uniformly for N participants
    initial_probabilities = np.random.uniform(0, 1, size=N)

    # Probit transformation
    probit_transformed = norm.ppf(initial_probabilities)

    # Hierarchical Normal Modeling
    group_mean = random_normal_truncate(0, 1, lower_bound=-6, upper_bound=6)
    group_std = np.random.uniform(0.1, 1) # weakly informative uniform prior

    # Model the probit-transformed values with a truncated normal distribution for each participant
    adjusted_p_tf = np.array([random_normal_truncate(group_mean, group_std, -6, 6) for _ in probit_transformed])

    return adjusted_p_tf


def simulate_trials_fixed_SSD(trial_type_sequence, ssd_set, p_tf,
                              mu_go, sigma_go, tau_go, 
                              mu_stop, sigma_stop, tau_stop):
    '''
    Simulate one synthetic experiment round of trials for a subject following the 
    pre-determined trial sequence (with fixed SSD). This simulation includes 
    both 'go' and 'stop' trials, with 'stop' trials having predetermined stop-signal delays (SSDs)
    from a specified set, presented in random order (with equal probability).

    Inputs:  
        1) trial_type_sequence: list of strings, where each string is either 'go' or 'stop',
           indicating the type of trial to simulate in sequence.
           Example: ['go', 'stop', 'go', 'stop', 'go']

        2) ssd_set: array or list of integers, specifying the set of SSDs (in milliseconds)
           to be used for the 'stop' trials. SSDs are assigned randomly to each 'stop' trial.
           Example: [100, 200, 300]

        3) p_tf: float, the probability of trigger failure (ranging from 0 to 1),
           where a trigger failure means the stop process is not even initiated.

        4) mu_go: float, the mean of the Gaussian component of the Ex-Gaussian distribution
           for the 'go' reaction times (RTs).

        5) sigma_go: float, the standard deviation of the Gaussian component of the 
           Ex-Gaussian distribution for the 'go' RTs.

        6) tau_go: float, the decay parameter (mean of the exponential component) of the
           Ex-Gaussian distribution for the 'go' RTs.

        7) mu_stop: float, the mean of the Gaussian component of the Ex-Gaussian distribution
           for the 'stop' reaction times (SSRTs).

        8) sigma_stop: float, the standard deviation of the Gaussian component of the 
           Ex-Gaussian distribution for the SSRTs.

        9) tau_stop: float, the decay parameter of the Ex-Gaussian distribution for the SSRTs.

    Returns:
        A pandas DataFrame containing the outcomes of each trial. Columns include:
        - 'trial_type': type of trial ('go' or 'stop').
        - 'ssd' (only for 'stop' trials): the stop-signal delay used in that trial.
        - 'observed_rt': the observed simulated reaction time (None for successful inhibitions in stop trials).
        - 'ss_rt': unobserved reaction time for stop signals (None for go trials).
        - 'outcome': describes the result of the trial ('go', 'stop-respond', or 'successful inhibition').
    '''
    results = []
    ssd_choices = np.random.choice(
        ssd_set, 
        size=len([t for t in trial_type_sequence if t == 'stop']), 
        replace=True
    )

    ssd_index = 0

    for t in trial_type_sequence:
        trial_data = {'trial_type': t, 'ssd': None, 'observed_rt': None, 'ss_rt': None, 'outcome': None}

        if t == 'stop':
            ssd = ssd_choices[ssd_index]
            ssd_index += 1

            go_rt = simulate_exgaussian(mu_go, sigma_go, tau_go)
            ssrt = simulate_exgaussian(mu_stop, sigma_stop, tau_stop)

            if random.random() < p_tf:
                trial_data.update({'ssd': ssd, 'observed_rt': go_rt, 'ss_rt': ssrt, 'outcome': 'stop-respond'})
            else:
                if go_rt > ssrt + ssd:
                    trial_data.update({'ssd': ssd, 'ss_rt': ssrt, 'outcome': 'successful inhibition'})
                else:
                    trial_data.update({'ssd': ssd, 'observed_rt': go_rt, 'ss_rt': ssrt, 'outcome': 'stop-respond'})
        else:
            go_rt = simulate_exgaussian(mu_go, sigma_go, tau_go)
            trial_data.update({'observed_rt': go_rt, 'outcome': 'go'})

        results.append(trial_data)

    return pd.DataFrame(results)

def simulate_trials_staircase_SSD(trial_type_sequence, starting_ssd, p_tf,
                                  mu_go, sigma_go, tau_go, 
                                  mu_stop, sigma_stop, tau_stop):
    '''
    Simulate synthetic experiment trials for a subject following the 
    pre-determined trial sequence using a staircase SSD procedure.

    Inputs:  
        1) trial_type_sequence: List of strings, each 'go' or 'stop' indicating the type of trial.
        2) starting_ssd: Initial SSD in milliseconds to start the staircase procedure.
        3) p_tf: Probability of trigger failure, where the stop signal fails to initiate.
        4) mu_go, sigma_go, tau_go: Parameters for the Ex-Gaussian distribution of go RTs.
        5) mu_stop, sigma_stop, tau_stop: Parameters for the Ex-Gaussian distribution of stop RTs.

    Returns:
        A pandas DataFrame containing the outcomes of each trial. Columns include:
        - 'trial_type': type of trial ('go' or 'stop').
        - 'ssd' (only for 'stop' trials): the stop-signal delay used in that trial.
        - 'observed_rt': the observed simulated reaction time (None for successful inhibitions in stop trials).
        - 'ss_rt': unobserved reaction time for stop signals (None for go trials).
        - 'outcome': describes the result of the trial ('go', 'stop-respond', or 'successful inhibition').
    '''
    results = []
    ssd = starting_ssd

    for t in trial_type_sequence:
        trial_data = {'trial_type': t, 'ssd': None, 'observed_rt': None, 'ss_rt': None, 'outcome': None}
        
        if t == 'stop':
            trial_data['ssd'] = ssd

            go_rt = simulate_exgaussian(mu_go, sigma_go, tau_go)
            ssrt = simulate_exgaussian(mu_stop, sigma_stop, tau_stop)
            
            if random.random() < p_tf:
                trial_data.update({'observed_rt': go_rt, 'ss_rt': ssrt, 'outcome': 'stop-respond'})
            else:
                if go_rt > ssrt + ssd:
                    trial_data.update({'ss_rt': ssrt, 'outcome': 'successful inhibition'})
                else:
                    trial_data.update({'observed_rt': go_rt, 'ss_rt': ssrt, 'outcome': 'stop-respond'})
            
            # Dynamically adjust ssd based on performance
            ssd = ssd + 50 if trial_data['outcome'] == 'successful inhibition' else ssd - 50
        else:
            go_rt = simulate_exgaussian(mu_go, sigma_go, tau_go)
            trial_data.update({'rt': go_rt, 'outcome': 'go'})

        results.append(trial_data)

    return pd.DataFrame(results)

'''
Functions related to likelihood
'''
# Reference paper for calculating the pdf and cdf of Ex-Gaussian distribution:
# Matzke, D., Dolan, C. V., Logan, G. D., Brown, S. D., & Wagenmakers, E. J. (2013). Bayesian parametric estimation of stop-signal reaction time distributions. Journal of Experimental Psychology: General, 142(4), 1047.
def exgaussian_pdf(x, mu, sigma, tau):
    """
    Probability Density Function (pdf) for Ex-Gaussian distribution using erf 
    (taking advanatge of numpy vectoized operations).

    Inputs:
        1) x: point(s) of which to find pdf (here, observed RT for go trials; could be scalars or numpy arrays)
        2) mu: mean of the Gaussian component
        3) sigma: standard deviation of the Gaussian component
        4) tau: mean (or rate parameter) of the exponential component
    
    Returns: 
        pdf for given x
    """
    
    lambd = 1 / tau
    exp_part = lambd * np.exp((lambd * (mu - x)) + (sigma * lambd)**2 / 2)
    norm_cdf_part = norm.cdf((x - mu) / sigma - lambd * sigma)
    return exp_part * norm_cdf_part

def exgaussian_cdf(x, mu, sigma, tau):
    """
    Cumulative Distribution Function (CDF) for Ex-Gaussian distribution using er
    (taking advanatge of numpy vectoized operations).

    Inputs:
        1) x: point of which to find cdf (here, observed RT for go trials; could be scalars or numpy arrays)
        2) mu: mean of the Gaussian component
        3) sigma: standard deviation of the Gaussian component
        4) tau: mean (or rate parameter) of the exponential component

    Returns: 
        cdf for given x
    """

    lambd = 1 / tau
    norm_cdf_part1 = norm.cdf((x - mu) / sigma)
    exp_part = np.exp((sigma * lambd)**2 / 2 - (x - mu) * lambd)
    norm_cdf_part2 = norm.cdf((x - mu) / sigma - sigma * lambd)
    return norm_cdf_part1 - exp_part * norm_cdf_part2

# Reference paper for calculating the likelihood functions of three types of RT
# Matzke, D., Love, J., & Heathcote, A. (2017). A Bayesian approach for estimating the probability of trigger failures in the stop-signal paradigm. Behavior research methods, 49, 267-281.
def go_RT_log_likelihood(t_g, mu_go, sigma_go, tau_go):
    '''
    Likelihood function for response times in go trials using the Ex-Gaussian distribution.

    Inputs:
        t_g: array of observed response times (go RT)
        mu_go: mean of the Gaussian component for go RT distribution
        sigma_go: standard deviation of the Gaussian component for go RT distribution
        tau_go: mean (or rate parameter) of the exponential component for go RT distribution
    
    Returns:
        Overall likelihood for the observed set of response times in go trials
    '''
    # Calculate the PDF for each response time (array of indiviudal trial likelihoods)
    individual_likelihoods = exgaussian_pdf(t_g, mu_go, sigma_go, tau_go)
    
    # Logarithms are used to avoid underflow in computation of very small numbers
    log_likelihoods = np.log(individual_likelihoods)
    total_log_likelihood = np.sum(log_likelihoods)
    
    return total_log_likelihood

def signal_respond_RT_log_likelihood(t_r, mu_go, sigma_go, tau_go, 
                                     mu_stop, sigma_stop, tau_stop, 
                                     p_tf, SSD):
    '''
    Likelihood function for signal-response response times in stop-response trials 
    using the Ex-Gaussian distribution.

    Inputs:
        t_r: array of observed response times (singal-respond RT)
        mu_go: mean of the Gaussian component for go RT distribution
        sigma_go: standard deviation of the Gaussian component for go RT distribution
        tau_go: mean (or rate parameter) of the exponential component for go RT distribution
        mu_stop: mean of the Gaussian component for stop RT distribution
        sigma_stop: standard deviation of the Gaussian component for stop RT distribution
        tau_stop: mean (or rate parameter) of the exponential component for stop RT distribution
        p_tf: probability of trigger failure (of stop signals)
        SSD: array of stop-signal delay time (for stop trials)
    
    Returns:
        Overall likelihood for the observed set of signal-response response times in stop-response trials 
    '''
    # Calculate the "part" of the likelihood related to trigger failure
    failed_trigger = p_tf * exgaussian_pdf(t_r, mu_go, sigma_go, tau_go)

    # Calculate the "part" of the likelihood when the stop process was 
    # successfully triggered but has finished after the go process 
    # (censored go RT distribution with SSD + SSRT being the right censoring point)
    adjusted_mu_stop = mu_stop + SSD # shifts the Ex-Gaussian distrbution by SSD
    successful_trigger = (1 - p_tf) * (1 - exgaussian_cdf(t_r, adjusted_mu_stop, sigma_stop, tau_stop)) * exgaussian_pdf(t_r, mu_go, sigma_go, tau_go)

    log_likelihoods = np.log(failed_trigger + successful_trigger)
    total_log_likelihood = np.sum(log_likelihoods)

    return total_log_likelihood

def gauss_legendre_quadrature(upper_bound, SSD, n=50):
    '''
    Generates the nodes and weights for Gauss-Legendre quadrature.
    '''
    # The upper bound here is the maximum observed response time plus 10 seconds 
    # to ensure that it covers the whole range of distribution (i.e., SSRT 
    # distrbution shifted by SSD).
    nodes, weights = np.polynomial.legendre.leggauss(n)

    # Transform nodes from [-1, 1] to [SSD, upper_bound]
    transformed_nodes = 0.5 * (nodes + 1) * (upper_bound - SSD) + SSD
    transformed_weights = weights * 0.5 * (upper_bound - SSD)

    return transformed_nodes, transformed_weights

def inhibit_log_likelihood_numeric_approximation(mu_go, sigma_go, tau_go, 
                                                 mu_stop, sigma_stop, tau_stop, 
                                                 p_tf, upper_bound, SSD, n=50):
    '''
    Likelihood function for successful inhition trials (in stop-response trials).
    Since the likelihood involves the intergration of (the composite of) pdf and
    cdf of ex-gaussian distributions (over t_i), it is not an analytical 
    likelihood function. The likelihood (approximation) function here uses 
    quadrature rules to numerically approximate the integral. Here, I implement 
    Gaussian quadrature (specifically Gauss-Legendre) over an interval defined 
    by 0 to the 95th percentile of the stop-signal 
    reaction time (SSRT) distribution. 

    Inputs:
        mu_go: mean of the Gaussian component for go RT distribution
        sigma_go: standard deviation of the Gaussian component for go RT distribution
        tau_go: mean (or rate parameter) of the exponential component for go RT distribution
        mu_stop: mean of the Gaussian component for stop RT distribution
        sigma_stop: standard deviation of the Gaussian component for stop RT distribution
        tau_stop: mean (or rate parameter) of the exponential component for stop RT distribution
        p_tf: probability of trigger failure (of stop signals)
        upper_bound: upper bound of ti in nuermical intergration
        SSD: array of stop-signal delay time (for stop trials)
        n: number of nodes for Gauss-Legendre quadrature
    
    Returns:
        Overall likelihood for the observed set of signal-response response times in stop-response trials 
    '''

    # Iterate through each successful inhibition trial to calculate likelihood for each trial 
    individual_likelihoods = np.zeros(len(SSD))
    
    for i, ssd in enumerate(SSD):
        # Derive the intergral in likelihood function for successful inhibition
        total_integral = 0
        
        # Retrieve nodes and weights for Gauss-Legendre quadrature
        nodes, weights = gauss_legendre_quadrature(upper_bound, ssd, n)

        for j in range(n):
            # Note: each node[j] means (unobservable) stop process finishing time
            intergrand = (1 - exgaussian_cdf(nodes[j], mu_go, sigma_go, tau_go)) * exgaussian_pdf(nodes[j], mu_stop + ssd, sigma_stop, tau_stop)
            integral = weights[j] * intergrand
            total_integral += integral
        
        # Derive likelihood for an individual trial
        individual_likelihoods[i] = (1 - p_tf) * total_integral

    # Calculate the overall likelihood
    log_likelihoods = np.log(individual_likelihoods)
    total_log_likelihood = np.sum(log_likelihoods)

    return total_log_likelihood
