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
    
    # K is the shape parameter for the exponential part
    K = tau / sigma
    # Simulate Ex-Gaussian distribution using scipy's exponnorm which is a combination of exponential and normal distributions
    simulated_value = exponnorm.rvs(K=K, loc=mu, scale=sigma, size=1)[0]
    
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