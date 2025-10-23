import numpy as np
import pandas as pd
import random
from scipy.stats import norm, exponnorm

'''
Functions related to simulation
'''
def simulate_exgaussian(mu, sigma, tau):
    """Generate a random value following Ex-Gaussian distribution with given parameters.

    Parameters
    ----------
    mu: mean of the Gaussian component.
    sigma: standard deviation of the Gaussian component.
    tau: mean (or rate parameter) of the exponential component.

    Returns
    -------
    A simulated value following Ex-Gaussian distribution with given parameters.
    """
    # K is the shape parameter for the exponential part
    K = tau / sigma

    # Simulate Ex-Gaussian distribution using scipy's exponnorm which 
    # is a combination of exponential and normal distributions
    while True:
        simulated_value = exponnorm.rvs(K=K, loc=mu, scale=sigma, size=1)[0]
        # Regenerate until a non-negative value is obtained
        if simulated_value >= 0:
            return simulated_value

def simulate_trials_fixed_SSD(trial_type_sequence, ssd_set, p_tf,
                              mu_go, sigma_go, tau_go, 
                              mu_stop, sigma_stop, tau_stop):
    """
    Simulate one synthetic experiment round of trials for a subject following the 
    pre-determined trial sequence (with fixed SSD). This simulation includes 
    both 'go' and 'stop' trials, with 'stop' trials having predetermined 
    stop-signal delays (SSDs) from a specified set, presented in random order 
    (with equal probability).

    Parameters
    ---------- 
    trial_type_sequence: 
        list of strings, where each string is either 'go' or 'stop', indicating 
        the type of trial to simulate in sequence. 
        Example: ['go', 'stop', 'go', 'stop', 'go']
    ssd_set: 
        array or list of integers, specifying the set of SSDs (in milliseconds)
        to be used for the 'stop' trials. SSDs are assigned randomly to each 
        'stop' trial. 
        Example: [100, 200, 300]
    p_tf: 
        probability of trigger failure (ranging from 0 to 1), where a trigger 
        failure means the stop process is not even initiated.
    mu_go: 
        mean of the Gaussian component of the Ex-Gaussian distribution
        for the 'go' reaction times (RTs).
    sigma_go: 
        standard deviation of the Gaussian component of the Ex-Gaussian 
        distribution for the 'go' RTs.
    tau_go: 
        decay parameter (mean of the exponential component) of the Ex-Gaussian 
        distribution for the 'go' RTs.
    mu_stop: 
        mean of the Gaussian component of the Ex-Gaussian distribution for 
        the 'stop' reaction times (SSRTs).
    sigma_stop:
        standard deviation of the Gaussian component of the Ex-Gaussian 
        distribution for the SSRTs.
    tau_stop: 
        decay parameter of the Ex-Gaussian distribution for the SSRTs.

    Returns
    -------
    A pandas DataFrame containing the outcomes of each trial. Columns include:
        - 'trial_type': type of trial ('go' or 'stop').
        - 'ssd' (only for 'stop' trials): the stop-signal delay used in that trial.
        - 'observed_rt': the observed simulated reaction time 
            (None for successful inhibitions in stop trials).
        - 'ss_rt': unobserved reaction time for stop signals 
            (None for go trials).
        - 'outcome': describes the result of the trial 
            ('go', 'stop-respond', or 'successful inhibition').
    """
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

def simulate_trials_fixed_SSD_no_p_tf(trial_type_sequence, ssd_set,
                                      mu_go, sigma_go, tau_go, 
                                      mu_stop, sigma_stop, tau_stop):
    """
    Simulate one synthetic experiment round of trials for a subject following the 
    pre-determined trial sequence (with fixed SSD). This simulation includes 
    both 'go' and 'stop' trials, with 'stop' trials having predetermined 
    stop-signal delays (SSDs) from a specified set, presented in random order 
    (with equal probability).

    Parameters
    ---------- 
    trial_type_sequence: 
        list of strings, where each string is either 'go' or 'stop', indicating 
        the type of trial to simulate in sequence. 
        Example: ['go', 'stop', 'go', 'stop', 'go']
    ssd_set: 
        array or list of integers, specifying the set of SSDs (in milliseconds)
        to be used for the 'stop' trials. SSDs are assigned randomly to each 
        'stop' trial. 
        Example: [100, 200, 300]
    mu_go: 
        mean of the Gaussian component of the Ex-Gaussian distribution
        for the 'go' reaction times (RTs).
    sigma_go: 
        standard deviation of the Gaussian component of the Ex-Gaussian 
        distribution for the 'go' RTs.
    tau_go: 
        decay parameter (mean of the exponential component) of the Ex-Gaussian 
        distribution for the 'go' RTs.
    mu_stop: 
        mean of the Gaussian component of the Ex-Gaussian distribution for 
        the 'stop' reaction times (SSRTs).
    sigma_stop:
        standard deviation of the Gaussian component of the Ex-Gaussian 
        distribution for the SSRTs.
    tau_stop: 
        decay parameter of the Ex-Gaussian distribution for the SSRTs.

    Returns
    -------
    A pandas DataFrame containing the outcomes of each trial. Columns include:
        - 'trial_type': type of trial ('go' or 'stop').
        - 'ssd' (only for 'stop' trials): the stop-signal delay used in that trial.
        - 'observed_rt': the observed simulated reaction time 
            (None for successful inhibitions in stop trials).
        - 'ss_rt': unobserved reaction time for stop signals 
            (None for go trials).
        - 'outcome': describes the result of the trial 
            ('go', 'stop-respond', or 'successful inhibition').
    """
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
    The staircase procedure adjusts the SSD to target approximately 50% 
    successful inhibitions, which is optimal for SSRT estimation.

    Parameters
    ---------- 
    trial_type_sequence: 
        List of strings, each 'go' or 'stop' indicating the type of trial.
    starting_ssd: 
        Initial SSD in milliseconds to start the staircase procedure.
    p_tf: 
        Probability of trigger failure, where the stop signal fails to initiate.
    mu_go: 
        mean of the Gaussian component of the Ex-Gaussian distribution
        for the 'go' reaction times (RTs).
    sigma_go: 
        standard deviation of the Gaussian component of the Ex-Gaussian 
        distribution for the 'go' RTs.
    tau_go: 
        decay parameter (mean of the exponential component) of the Ex-Gaussian 
        distribution for the 'go' RTs.
    mu_stop: 
        mean of the Gaussian component of the Ex-Gaussian distribution for 
        the 'stop' reaction times (SSRTs).
    sigma_stop:
        standard deviation of the Gaussian component of the Ex-Gaussian 
        distribution for the SSRTs.
    tau_stop: 
        decay parameter of the Ex-Gaussian distribution for the SSRTs.

    Returns
    -------
    A pandas DataFrame containing the outcomes of each trial. Columns include:
        - 'trial_type': type of trial ('go' or 'stop').
        - 'ssd' (only for 'stop' trials): the stop-signal delay used in that trial.
        - 'observed_rt': the observed simulated reaction time 
            (None for successful inhibitions in stop trials).
        - 'ss_rt': unobserved reaction time for stop signals 
            (None for go trials).
        - 'outcome': describes the result of the trial 
            ('go', 'stop-respond', or 'successful inhibition').
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
            # Increase SSD by 50ms after successful inhibition, decrease by 50ms after failed inhibition
            # This aims to converge to ~50% successful inhibitions
            ssd = max(0, ssd + 50 if trial_data['outcome'] == 'successful inhibition' else ssd - 50)
        else:
            go_rt = simulate_exgaussian(mu_go, sigma_go, tau_go)
            trial_data.update({'observed_rt': go_rt, 'outcome': 'go'})

        results.append(trial_data)

    return pd.DataFrame(results)

def simulate_trials_staircase_SSD_no_p_tf(trial_type_sequence, starting_ssd,
                                          mu_go, sigma_go, tau_go, 
                                          mu_stop, sigma_stop, tau_stop):
    '''
    Simulate synthetic experiment trials for a subject following the 
    pre-determined trial sequence using a staircase SSD procedure without
    considering the probability of trigger failure (p_tf).
    The staircase procedure adjusts the SSD to target approximately 50% 
    successful inhibitions, which is optimal for SSRT estimation.

    Parameters
    ---------- 
    trial_type_sequence: 
        List of strings, each 'go' or 'stop' indicating the type of trial.
    starting_ssd: 
        Initial SSD in milliseconds to start the staircase procedure.
    mu_go: 
        Mean of the Gaussian component of the Ex-Gaussian distribution
        for the 'go' reaction times (RTs).
    sigma_go: 
        Standard deviation of the Gaussian component of the Ex-Gaussian 
        distribution for the 'go' RTs.
    tau_go: 
        Decay parameter (mean of the exponential component) of the Ex-Gaussian 
        distribution for the 'go' RTs.
    mu_stop: 
        Mean of the Gaussian component of the Ex-Gaussian distribution for 
        the 'stop' reaction times (SSRTs).
    sigma_stop:
        Standard deviation of the Gaussian component of the Ex-Gaussian 
        distribution for the SSRTs.
    tau_stop: 
        Decay parameter of the Ex-Gaussian distribution for the SSRTs.

    Returns
    -------
    A pandas DataFrame containing the outcomes of each trial. Columns include:
        - 'trial_type': type of trial ('go' or 'stop').
        - 'ssd' (only for 'stop' trials): the stop-signal delay used in that trial.
        - 'observed_rt': the observed simulated reaction time 
            (None for successful inhibitions in stop trials).
        - 'ss_rt': unobserved reaction time for stop signals 
            (None for go trials).
        - 'outcome': describes the result of the trial 
            ('go', 'stop-respond', or 'successful inhibition').
    '''
    results = []
    ssd = starting_ssd

    for t in trial_type_sequence:
        trial_data = {'trial_type': t, 'ssd': None, 'observed_rt': None, 'ss_rt': None, 'outcome': None}
        
        if t == 'stop':
            trial_data['ssd'] = ssd

            go_rt = simulate_exgaussian(mu_go, sigma_go, tau_go)
            ssrt = simulate_exgaussian(mu_stop, sigma_stop, tau_stop)
            
            # Determine outcome without trigger failure probability
            if go_rt > ssrt + ssd:
                trial_data.update({'ss_rt': ssrt, 'outcome': 'successful inhibition'})
            else:
                trial_data.update({'observed_rt': go_rt, 'ss_rt': ssrt, 'outcome': 'stop-respond'})
            
            # Dynamically adjust ssd based on performance
            # Increase SSD by 50ms after successful inhibition, decrease by 50ms after failed inhibition
            # This aims to converge to ~50% successful inhibitions
            ssd = max(0, ssd + 50 if trial_data['outcome'] == 'successful inhibition' else ssd - 50)
        else:
            go_rt = simulate_exgaussian(mu_go, sigma_go, tau_go)
            trial_data.update({'observed_rt': go_rt, 'outcome': 'go'})

        results.append(trial_data)

    return pd.DataFrame(results)

'''
Functions related to likelihood (written in numpy and scipy)
'''
# Reference paper for calculating the pdf and cdf of Ex-Gaussian distribution:
    # Matzke, D., Dolan, C. V., Logan, G. D., Brown, S. D., & Wagenmakers, E. J. (2013). 
    # Bayesian parametric estimation of stop-signal reaction time distributions. 
    # Journal of Experimental Psychology: General, 142(4), 1047.
def exgaussian_pdf(x, mu, sigma, tau):
    """
    Probability Density Function (pdf) for Ex-Gaussian distribution using erf 
    (taking advanatge of numpy vectoized operations).

    Parameters
    ---------- 
    x: point(s) of which to find pdf.
    mu: mean of the Gaussian component.
    sigma: standard deviation of the Gaussian component.
    tau: mean (or rate parameter) of the exponential component.
    
    Returns
    ------- 
    exgaussian pdf for given x.
    """
    lambd = 1 / tau
    exp_part = lambd * np.exp((lambd * (mu - x)) + (sigma * lambd)**2 / 2)
    norm_cdf_part = norm.cdf((x - mu) / sigma - lambd * sigma)
    return exp_part * norm_cdf_part

def exgaussian_cdf(x, mu, sigma, tau):
    """
    Cumulative Distribution Function (CDF) for Ex-Gaussian distribution using er
    (taking advanatge of numpy vectoized operations).

    Parameters
    ----------
    x: point of which to find cdf.
    mu: mean of the Gaussian component.
    sigma: standard deviation of the Gaussian component.
    tau: mean (or rate parameter) of the exponential component.

    Returns
    -------
    exgaussian cdf for given x.
    """
    lambd = 1 / tau
    norm_cdf_part1 = norm.cdf((x - mu) / sigma)
    exp_part = np.exp((sigma * lambd)**2 / 2 - (x - mu) * lambd)
    norm_cdf_part2 = norm.cdf((x - mu) / sigma - sigma * lambd)
    return norm_cdf_part1 - exp_part * norm_cdf_part2

# Reference paper for calculating the likelihood functions of three types of RT
    # Matzke, D., Love, J., & Heathcote, A. (2017). A Bayesian approach for 
    # estimating the probability of trigger failures in the stop-signal paradigm. 
    # Behavior research methods, 49, 267-281.
def gauss_legendre_quadrature(upper_bound, SSD, n=50):
    '''
    Generate the nodes (sample points) and weights for Gauss-Legendre 
    quadrature, which is a numerical integration method. The nodes and weights 
    are transformed to fit the interval [SSD, upper_bound].

    Parameters
    ----------
    upper_bound: 
        The upper limit for the interval of integration. Typically, this is the 
        maximum observed response time plus an additional buffer 
        (e.g., 10 seconds) to ensure the entire distribution is covered.
    SSD: 
        The stop-signal delay, which serves as the lower limit for the 
        interval of integration.
    n: 
        The number of nodes and weights to generate. The default is 50.

    Returns
    -------
    transformed_nodes:
        The nodes (sample points) for the Gauss-Legendre quadrature, 
        transformed to the interval [SSD, upper_bound].
    transformed_weights: 
        The weights for the Gauss-Legendre quadrature, 
        corresponding to the transformed nodes.
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
                                                 p_tf, upper_bound, ssd_array, 
                                                 n=50):
    '''
    Likelihood function for successful inhition trials (in stop-response trials).
    Since the likelihood involves the intergration of (the composite of) pdf and
    cdf of ex-gaussian distributions (over t_i), it is not an analytical 
    likelihood function. The likelihood (approximation) function here uses 
    quadrature rules to numerically approximate the integral. Here, I implement 
    Gaussian quadrature (specifically Gauss-Legendre) over an interval defined 
    by 0 to the 95th percentile of the stop-signal 
    reaction time (SSRT) distribution. 

    Parameters
    ----------
    mu_go: 
        mean of the Gaussian component for go RT distribution.
    sigma_go: 
        standard deviation of the Gaussian component for go RT distribution.
    tau_go: 
        mean (or rate parameter) of the exponential component for go RT distribution.
    mu_stop: 
        mean of the Gaussian component for stop RT distribution.
    sigma_stop: 
        standard deviation of the Gaussian component for stop RT distribution.
    tau_stop: 
        mean (or rate parameter) of the exponential component for stop RT distribution.
    p_tf: 
        probability of trigger failure (of stop signals).
    upper_bound: 
        upper bound of ti in nuermical intergration.
    ssd_array: 
        array of stop-signal delay time (for stop trials).
    n: 
        number of nodes for Gauss-Legendre quadrature.
    
    Returns
    -------
    Overall likelihood for the observed set of signal-response response 
    times in stop-response trials.
    '''

    individual_likelihoods = []

    for ssd in ssd_array:
        total_integral = 0
        nodes, weights = gauss_legendre_quadrature(upper_bound, ssd, n)

        for j in range(n):
            integrand = (1 - exgaussian_cdf(nodes[j], mu_go, sigma_go, tau_go)) * exgaussian_pdf(nodes[j] - ssd, mu_stop, sigma_stop, tau_stop)
            integral = weights[j] * integrand
            total_integral += integral

        individual_likelihood = (1 - p_tf) * total_integral
        individual_likelihoods.append(individual_likelihood)

    log_likelihoods = np.log(individual_likelihoods)
    total_log_likelihood = np.sum(log_likelihoods)

    return total_log_likelihood