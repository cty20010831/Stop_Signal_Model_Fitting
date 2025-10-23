'''
This python contains likelihood-related functions for pymc model fitting/sampling.
Most of these (pymc) mathematical functions are imported directly from pytensor.tensor.
Doing any kind of math with PyMC random variables, or defining custom likelihoods 
or priors requires using PyTensor expressions rather than NumPy or Python code.

Note:
There is an earlier version where I tried the other integration method (using 
`lqag` from `lintegrate` => https://github.com/mattpitkin/lintegrate). However, 
there were some issues when running the code, returning errors that I 
unfortunately could not debug.
'''
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import arviz as az
import pymc as pm
from pymc.math import logdiffexp
from pymc.distributions.dist_math import normal_lcdf, log_normal, check_parameters
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os

# Dynamically append the parent directory of the current file's directory to the path (import from simulation)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulation.simSST import simulate_participant_blocks

# alternatives to pt.switch 
# logpdf with exgaussian plot
def logpdf_exgaussian(value, mu, sigma, tau):
    res = pt.switch(
            # Whether tau > 0.05 * sigma
            pt.gt(tau, 0.05 * sigma),
            # If True
            (
                -pt.log(tau)
                + (mu - value) / tau
                + 0.5 * (sigma / tau) ** 2
                + normal_lcdf(mu + (sigma**2) / tau, sigma, value)
            ),
            # If False
            log_normal(value, mean=mu, sigma=sigma),
        )
        
    return check_parameters(
        res,
        sigma > 0,
        tau > 0,
        msg="nu > 0, sigma > 0",
    )

def logcdf_exgaussian(value, mu, sigma, tau):
    res = pt.switch(
        # Whether tau > 0.05 * sigma
        pt.gt(tau, 0.05 * sigma),
        # If True
        logdiffexp(normal_lcdf(mu, sigma, value),
                        (
                            (mu - value) / tau
                            + 0.5 * (sigma / tau) ** 2
                            + normal_lcdf(mu + (sigma**2) / tau, sigma, value)
                        ),
                    ),
        # If False
        normal_lcdf(mu, sigma, value)
    )

    return check_parameters(
            res,
            sigma > 0,
            tau > 0,
            msg="sigma > 0, nu > 0",
        )

def stop_respond_log_likelihood(t_r, mu_go, sigma_go, tau_go, 
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

    Returns
    -------
    pytensor.tensor
        Log-likelihood of the response times given the parameters.
    """
    # Derive the Ex-Gaussian PDF and CDF using PyMC functions
    exgaussian_go = pm.ExGaussian.dist(mu=mu_go, sigma=sigma_go, nu=tau_go)
    exgaussian_stop = pm.ExGaussian.dist(mu=mu_stop, sigma=sigma_stop, nu=tau_stop)
    
    # failed_trigger = p_tf * pt.exp(pt.exp(pm.logp(exgaussian_go, t_r)))
    failed_trigger = p_tf * pt.exp(logpdf_exgaussian(t_r, mu_go, sigma_go, tau_go))
    
    # Focus on the relevant time interval for determining if the stop process 
    # finished before the response was made (specifically, t_r - ssd_array)
    # successful_trigger = (1 - p_tf) * (1 - pt.exp(pm.logcdf(exgaussian_stop, t_r - ssd))) * pt.exp(pm.logp(exgaussian_go, t_r))

    # Original implementation following the paper
    successful_trigger = (1 - p_tf) * (1 - pt.exp(logcdf_exgaussian(t_r - ssd, mu_stop, sigma_stop, tau_stop))) * pt.exp(logpdf_exgaussian(t_r, mu_go, sigma_go, tau_go)) 
    # BEESTS implementation (see BEESTS/BEESTS_Windows/stopsignal/src/stop_likelihoods_wtf.pyx)
    # successful_trigger = (1 - p_tf) * (pt.exp(logcdf_exgaussian(t_r - ssd, mu_stop, sigma_stop, tau_stop))) * pt.exp(logpdf_exgaussian(t_r, mu_go, sigma_go, tau_go)) 
    
    likelihood = failed_trigger + successful_trigger
    
    return pt.log(likelihood)

def precompute_legendre_quadrature(ssd, upper_bound, n):
    """
    Precompute Gauss-Legendre quadrature nodes and weights for numerical integration.

    Parameters
    ----------
    ssd : array-like
        Array of stop-signal delays (SSD) for each successful inhibition trial.
    upper_bound : array-like
        Upper bound(s) of response time. Can be a single value applied to all participants
        or an array with an upper bound for each participant.
    n : int
        Number of Gauss-Legendre quadrature points.

    Returns
    -------
    transformed_nodes : numpy.ndarray
        Transformed nodes for Gauss-Legendre quadrature integration.
    transformed_weights : numpy.ndarray
        Transformed weights for Gauss-Legendre quadrature integration.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)
    
    if len(upper_bound) == 1:
        # Deal with the case when there is only one upper bound for all participants
        # Transform nodes from [-1, 1] to [lower_bound, upper_bound]
        lower_bound = np.array(ssd)[None, :]  # Expand dims to (1, len(ssd))
        upper_bound = np.array(upper_bound)[:, None]  # Expand dims to (len(upper_bound), 1)
    else:
        # Deal with the case when there is an upper_bound of response time for each participant
        # Transform nodes from [-1, 1] to [lower_bound, upper_bound]
        lower_bound = np.array(ssd)
        upper_bound = np.array(upper_bound)

    transformed_nodes = 0.5 * (nodes[:, None] + 1) * (upper_bound - lower_bound) + lower_bound
    transformed_weights = weights[:, None] * 0.5 * (upper_bound - lower_bound)

    return transformed_nodes, transformed_weights

def integrate_cexgauss(nodes, weights, mu_go, sigma_go, tau_go, mu_stop, sigma_stop, tau_stop, ssd):
    """
    Perform numerical integration of the Censored Ex-Gaussian using Gauss-Legendre quadrature.

    Parameters
    ----------
    nodes : array-like
        Transformed nodes for Gauss-Legendre quadrature rule calculation.
    weights : array-like
        Transformed weights for Gauss-Legendre quadrature rule calculation.
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
    ssd : array-like
        Stop-signal delays.

    Returns
    -------
    pytensor.tensor
        Result of the numerical integration for the Censored Ex-Gaussian.
    """
    # Derive the Ex-Gaussian PDF and CDF using PyMC functions
    exgaussian_go = pm.ExGaussian.dist(mu=mu_go, sigma=sigma_go, nu=tau_go)
    exgaussian_stop = pm.ExGaussian.dist(mu=mu_stop, sigma=sigma_stop, nu=tau_stop)

    exgaussian_cdf_vals = pt.exp(logcdf_exgaussian(nodes, mu_go, sigma_go, tau_go))
    exgaussian_pdf_vals = pt.exp(logpdf_exgaussian(nodes - ssd, mu_stop, sigma_stop, tau_stop))

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

    # The following commented-out part was used to count each respective ssd for 
    # each participant and caluclate there respective integral only once 
    # so that calculating the sum of log-likelihoods will be more efficient

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

def summary_stats(trace, participant_parameters=['mu_go', 'sigma_go', 'tau_go', 'mu_stop', 'sigma_stop', 'tau_stop', 'p_tf']):
    '''
    Extract summary statistics from pymc trace.
    Parameters  
    ----------
        trace: InferenceData
            trace from pymc model fitting   
        participant_parameters: list
            a list of parameter names to extract from the trace
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the summary statistics for each participant. 
        The DataFrame has columns for participant index and each parameter. 
    '''
    
    # Convert pymc estimate to a DataFrame
    pymc_estimate = pd.DataFrame(az.summary(trace, var_names=participant_parameters)[['mean']]).reset_index()
    pymc_estimate.columns=['parameter', 'value']

    # Extract the subject index and parameter name from 'parameter' column
    pymc_estimate['subj_idx'] = pymc_estimate['parameter'].str.extract(r'\[(\d+)\]').astype(int) + 1
    pymc_estimate['parameter'] = pymc_estimate['parameter'].str.extract(r'([a-zA-Z_]+)')

    # Pivot the table to achieve the desired format
    pymc_estimate = pymc_estimate.pivot(index='subj_idx', columns='parameter', values='value').reset_index()
    pymc_estimate.columns.name = None

    # Reorder columns as required
    pymc_estimate = pymc_estimate[['subj_idx'] + participant_parameters]

    return pymc_estimate

def posterior_predictive_sampling(trace,
                                  n_trials_per_block, prop_stop=0.25,
                                  n_blocks=1,
                                  observed_data=None, comparison_trace=None,
                                  step_size=50, ssd_start=None,
                                  plot=False):
    '''
    Conduct posterior predictive check with randomly selected parameter vectors,
    or cross-validation between two likelihood implementations.

    Uses staircase stop-signal task simulation with trigger failures (p_tf).
    Trials are randomly shuffled to maintain the go/stop ratio (no fixed sequence).
    SSD tracking is continuous across all blocks (not reset between blocks).

    Parameters
    ----------
        trace: InferenceData
            trace from pymc model fitting (primary trace)
        n_trials_per_block: int
            total number of trials per simulation block
        prop_stop: float, optional
            proportion of stop trials (default: 0.25 for 25% stop trials)
        n_blocks: int, optional
            number of blocks to simulate per participant (default: 1)
            SSD is tracked continuously across all blocks
        observed_data: pd.DataFrame or None
            observed go RT, signal-respond rate, and SSRT for comparison.
            Required when comparison_trace is None (traditional PPC mode).
        comparison_trace: InferenceData or None
            trace from alternative likelihood implementation for cross-validation.
            If provided, compares simulations from trace vs. comparison_trace instead
            of comparing to observed_data.
        step_size: float, optional
            staircase increment/decrement in ms (default: 50)
        ssd_start: float or None, optional
            starting SSD. If None, randomly picks from [150, 200, 250, 300] (default: None)
        plot: boolean
            whether to plot observed vs predicted distributions (or comparison distributions)

    Returns
    -------
    dict
        Simulated data and comparison metrics.
        - Traditional PPC mode: returns p-values comparing simulations to observed data
        - Cross-validation mode: returns metrics comparing two implementations
    '''
    # Validation
    if comparison_trace is None and observed_data is None:
        raise ValueError("Either observed_data (for PPC) or comparison_trace (for cross-validation) must be provided.")

    cross_validation_mode = comparison_trace is not None

    # Extract posterior samples from primary trace
    n_chains, n_draws, n_participants = trace.posterior['mu_go'].shape

    # Flatten the samples to have shape (n_chain * n_draws, n_participants)
    mu_go_samples = trace.posterior['mu_go'].values.reshape(-1, n_participants)
    sigma_go_samples = trace.posterior['sigma_go'].values.reshape(-1, n_participants)
    tau_go_samples = trace.posterior['tau_go'].values.reshape(-1, n_participants)
    mu_stop_samples = trace.posterior['mu_stop'].values.reshape(-1, n_participants)
    sigma_stop_samples = trace.posterior['sigma_stop'].values.reshape(-1, n_participants)
    tau_stop_samples = trace.posterior['tau_stop'].values.reshape(-1, n_participants)
    p_tf_samples = trace.posterior['p_tf'].values.reshape(-1, n_participants)

    # Extract posterior samples from comparison trace if in cross-validation mode
    if cross_validation_mode:
        mu_go_samples_comp = comparison_trace.posterior['mu_go'].values.reshape(-1, n_participants)
        sigma_go_samples_comp = comparison_trace.posterior['sigma_go'].values.reshape(-1, n_participants)
        tau_go_samples_comp = comparison_trace.posterior['tau_go'].values.reshape(-1, n_participants)
        mu_stop_samples_comp = comparison_trace.posterior['mu_stop'].values.reshape(-1, n_participants)
        sigma_stop_samples_comp = comparison_trace.posterior['sigma_stop'].values.reshape(-1, n_participants)
        tau_stop_samples_comp = comparison_trace.posterior['tau_stop'].values.reshape(-1, n_participants)
        p_tf_samples_comp = comparison_trace.posterior['p_tf'].values.reshape(-1, n_participants)

    # Randomly select 1,000 parameter vectors
    idx_samples = np.random.choice(mu_go_samples.shape[0], 1000, replace=False)

    all_simulated_trials = []
    all_simulated_trials_comp = []  # For comparison trace
    p_values_go = []
    p_values_sr = []
    p_values_sr_rt = []  # For signal-respond RT
    signal_respond_rts = []

    # Cross-validation metrics
    diff_go_rt_median = []
    diff_sr_rate = []
    diff_sr_rt_median = []

    for i in idx_samples:
        for participant in range(n_participants):
            # Simulate from primary trace
            mu_go, sigma_go, tau_go = mu_go_samples[i, participant], sigma_go_samples[i, participant], tau_go_samples[i, participant]
            mu_stop, sigma_stop, tau_stop = mu_stop_samples[i, participant], sigma_stop_samples[i, participant], tau_stop_samples[i, participant]
            p_tf = p_tf_samples[i, participant]

            simulated_trials = simulate_participant_blocks(
                n_blocks=n_blocks,
                n_trials_per_block=n_trials_per_block,
                prop_stop=prop_stop,
                mu_go=mu_go,
                sigma_go=sigma_go,
                tau_go=tau_go,
                mu_stop=mu_stop,
                sigma_stop=sigma_stop,
                tau_stop=tau_stop,
                p_tf=p_tf,
                ssd_start=ssd_start,
                step_size=step_size,
                include_block_id=True
            )
            simulated_trials['participant_id'] = participant
            all_simulated_trials.append(simulated_trials)

            # Simulate from comparison trace if in cross-validation mode
            if cross_validation_mode:
                mu_go_comp, sigma_go_comp, tau_go_comp = mu_go_samples_comp[i, participant], sigma_go_samples_comp[i, participant], tau_go_samples_comp[i, participant]
                mu_stop_comp, sigma_stop_comp, tau_stop_comp = mu_stop_samples_comp[i, participant], sigma_stop_samples_comp[i, participant], tau_stop_samples_comp[i, participant]
                p_tf_comp = p_tf_samples_comp[i, participant]

                simulated_trials_comp = simulate_participant_blocks(
                    n_blocks=n_blocks,
                    n_trials_per_block=n_trials_per_block,
                    prop_stop=prop_stop,
                    mu_go=mu_go_comp,
                    sigma_go=sigma_go_comp,
                    tau_go=tau_go_comp,
                    mu_stop=mu_stop_comp,
                    sigma_stop=sigma_stop_comp,
                    tau_stop=tau_stop_comp,
                    p_tf=p_tf_comp,
                    ssd_start=ssd_start,
                    step_size=step_size,
                    include_block_id=True
                )
                simulated_trials_comp['participant_id'] = participant
                all_simulated_trials_comp.append(simulated_trials_comp)

            # Compute metrics based on mode
            if cross_validation_mode:
                # Compare primary vs. comparison simulations
                go_rts_primary = simulated_trials.loc[simulated_trials['trial_type'] == 'go', 'observed_rt']
                go_rts_comp = simulated_trials_comp.loc[simulated_trials_comp['trial_type'] == 'go', 'observed_rt']
                diff_go_rt_median.append(np.median(go_rts_primary) - np.median(go_rts_comp))

                # Compare signal-respond rates
                ssds = simulated_trials.loc[simulated_trials['trial_type'] == 'stop', 'ssd'].unique()
                for ssd in ssds:
                    rr_primary = np.mean(simulated_trials[(simulated_trials['trial_type'] == 'stop') & (simulated_trials['ssd'] == ssd)]['outcome'] == 'stop-respond')
                    rr_comp = np.mean(simulated_trials_comp[(simulated_trials_comp['trial_type'] == 'stop') & (simulated_trials_comp['ssd'] == ssd)]['outcome'] == 'stop-respond')
                    diff_sr_rate.append(rr_primary - rr_comp)

                # Compare signal-respond RTs
                sr_rts_primary = simulated_trials.loc[simulated_trials['outcome'] == 'stop-respond', 'observed_rt']
                sr_rts_comp = simulated_trials_comp.loc[simulated_trials_comp['outcome'] == 'stop-respond', 'observed_rt']
                if len(sr_rts_primary) > 0 and len(sr_rts_comp) > 0:
                    diff_sr_rt_median.append(np.median(sr_rts_primary) - np.median(sr_rts_comp))
            else:
                # Traditional PPC: Compare to observed data
                go_rts = simulated_trials.loc[simulated_trials['trial_type'] == 'go', 'observed_rt']
                obs_go_rts = observed_data.loc[observed_data['trial_type'] == "go", 'observed_rt']
                p_values_go.append(np.mean(go_rts > obs_go_rts))

                # Calculate Signal-Respond Rate p-values for each SSD
                ssds = observed_data.loc[observed_data['trial_type'] == 'stop', 'ssd'].unique()
                observed_rr = [
                    np.mean(observed_data[(observed_data['trial_type'] == 'stop') & (observed_data['ssd'] == ssd)]['outcome'] == 'stop-respond')
                    for ssd in ssds
                ]

                predicted_rrs = [
                    np.mean(simulated_trials[(simulated_trials['trial_type'] == 'stop') & (simulated_trials['ssd'] == ssd)]['outcome'] == 'stop-respond')
                    for ssd in ssds
                ]
                p_values_sr.extend([np.mean(pred >= obs) for pred, obs in zip(predicted_rrs, observed_rr)])

                # Calculate Signal-Respond RT p-values
                sr_rts_pred = simulated_trials.loc[simulated_trials['outcome'] == 'stop-respond', 'observed_rt']
                sr_rts_obs = observed_data.loc[observed_data['outcome'] == 'stop-respond', 'observed_rt']
                if len(sr_rts_pred) > 0 and len(sr_rts_obs) > 0:
                    p_values_sr_rt.append(np.mean(sr_rts_pred > sr_rts_obs))

    # Visualization for each participant
    if plot:
        if cross_validation_mode:
            # Cross-validation plotting: compare primary vs. comparison
            # n_participants was already extracted from trace.posterior['mu_go'].shape above
            fig, axes = plt.subplots(n_participants, 3, figsize=(21, n_participants * 4))

            # Ensure axes is always 2D (handle single participant case)
            if n_participants == 1:
                axes = axes.reshape(1, -1)

            for i in range(n_participants):
                participant_simulated_trials = [trial_df[trial_df['participant_id'] == i] for trial_df in all_simulated_trials]
                participant_simulated_trials_comp = [trial_df[trial_df['participant_id'] == i] for trial_df in all_simulated_trials_comp]

                # Plot 1: Go RT - Compare Primary vs. Comparison
                combined_go_rt_primary = np.concatenate([trial_df.loc[trial_df['trial_type'] == 'go', 'observed_rt'].values for trial_df in participant_simulated_trials])
                combined_go_rt_comp = np.concatenate([trial_df.loc[trial_df['trial_type'] == 'go', 'observed_rt'].values for trial_df in participant_simulated_trials_comp])

                sns.kdeplot(combined_go_rt_primary, color="blue", alpha=0.5, ax=axes[i, 0], label='Primary (JAX)')
                sns.kdeplot(combined_go_rt_comp, color="red", alpha=0.5, ax=axes[i, 0], label='Comparison (Cython)')

                median_primary = np.median(combined_go_rt_primary)
                median_comp = np.median(combined_go_rt_comp)
                axes[i, 0].axvline(median_primary, color='blue', linestyle='--', linewidth=2, label='Primary Median')
                axes[i, 0].axvline(median_comp, color='red', linestyle='--', linewidth=2, label='Comparison Median')

                axes[i, 0].set_title(f"Go RT - Participant {i + 1}")
                axes[i, 0].set_xlabel("RT (ms)")
                axes[i, 0].set_ylabel("Density")
                axes[i, 0].legend()

                # Plot 2: Signal-Respond RT - Compare Primary vs. Comparison
                sr_rts_primary = np.concatenate([trial_df.loc[trial_df['outcome'] == 'stop-respond', 'observed_rt'].values for trial_df in participant_simulated_trials if len(trial_df.loc[trial_df['outcome'] == 'stop-respond']) > 0])
                sr_rts_comp = np.concatenate([trial_df.loc[trial_df['outcome'] == 'stop-respond', 'observed_rt'].values for trial_df in participant_simulated_trials_comp if len(trial_df.loc[trial_df['outcome'] == 'stop-respond']) > 0])

                if len(sr_rts_primary) > 0 and len(sr_rts_comp) > 0:
                    sns.kdeplot(sr_rts_primary, color="blue", alpha=0.5, ax=axes[i, 1], label='Primary (JAX)')
                    sns.kdeplot(sr_rts_comp, color="red", alpha=0.5, ax=axes[i, 1], label='Comparison (Cython)')

                    median_sr_primary = np.median(sr_rts_primary)
                    median_sr_comp = np.median(sr_rts_comp)
                    axes[i, 1].axvline(median_sr_primary, color='blue', linestyle='--', linewidth=2, label='Primary Median')
                    axes[i, 1].axvline(median_sr_comp, color='red', linestyle='--', linewidth=2, label='Comparison Median')

                axes[i, 1].set_title(f"Signal-Respond RT - Participant {i + 1}")
                axes[i, 1].set_xlabel("RT (ms)")
                axes[i, 1].set_ylabel("Density")
                axes[i, 1].legend()

                # Plot 3: Signal-Respond Rate by SSD - Compare Primary vs. Comparison
                # Concatenate all stop trials from all simulations (like Go RT plotting)
                combined_stop_trials_primary = pd.concat([trial_df[trial_df['trial_type'] == 'stop'] for trial_df in participant_simulated_trials], ignore_index=True)
                combined_stop_trials_comp = pd.concat([trial_df[trial_df['trial_type'] == 'stop'] for trial_df in participant_simulated_trials_comp], ignore_index=True)

                # Get unique SSDs from the combined data
                ssds_list = sorted(combined_stop_trials_primary['ssd'].unique())

                rr_primary_by_ssd = []
                rr_comp_by_ssd = []
                for ssd in ssds_list:
                    # Calculate response rate for this SSD from combined data
                    primary_ssd_trials = combined_stop_trials_primary[combined_stop_trials_primary['ssd'] == ssd]
                    comp_ssd_trials = combined_stop_trials_comp[combined_stop_trials_comp['ssd'] == ssd]

                    rr_primary = np.mean(primary_ssd_trials['outcome'] == 'stop-respond')
                    rr_comp = np.mean(comp_ssd_trials['outcome'] == 'stop-respond')

                    rr_primary_by_ssd.append(rr_primary)
                    rr_comp_by_ssd.append(rr_comp)

                axes[i, 2].plot(range(len(ssds_list)), rr_primary_by_ssd, 'bo-', linewidth=2, label='Primary (JAX)')
                axes[i, 2].plot(range(len(ssds_list)), rr_comp_by_ssd, 'ro-', linewidth=2, label='Comparison (Cython)')
                axes[i, 2].set_xticks(range(len(ssds_list)))
                axes[i, 2].set_xticklabels([int(round(ssd)) for ssd in ssds_list])
                axes[i, 2].set_title(f"Inhibition Function - Participant {i + 1}")
                axes[i, 2].set_xlabel("SSD (ms)")
                axes[i, 2].set_ylabel("P(respond | stop signal)")
                axes[i, 2].legend()

            plt.tight_layout()
            plt.show()

        else:
            # Traditional PPC plotting: compare simulated vs. observed
            # n_participants was already extracted from trace.posterior['mu_go'].shape above
            fig, axes = plt.subplots(n_participants, 3, figsize=(21, n_participants * 4))

            # Ensure axes is always 2D (handle single participant case)
            if n_participants == 1:
                axes = axes.reshape(1, -1)

            for i in range(n_participants):
                participant_data = observed_data[observed_data['participant_id'] == i]
                participant_data.loc[:, 'observed_rt'] = pd.to_numeric(participant_data['observed_rt'])
                participant_simulated_trials = [trial_df[trial_df['participant_id'] == i] for trial_df in all_simulated_trials]

                # Plot 1: Histogram + KDE of Observed Go RTs with Predicted Distributions for Each Participant
                sns.histplot(participant_data.loc[participant_data['trial_type'] == "go", 'observed_rt'], bins=10, color="black", ax=axes[i, 0], kde=True, alpha=0.1, label='Observed Go RT')
                combined_go_rt = np.concatenate([trial_df.loc[trial_df['trial_type'] == 'go', 'observed_rt'].values for trial_df in participant_simulated_trials])
                sns.kdeplot(combined_go_rt, color="gray", alpha=0.5, ax=axes[i, 0], label='Predicted Go RT')

                # Calculate medians and plot as dashed lines
                observed_median_go_rt = np.median(participant_data.loc[participant_data['trial_type'] == 'go', 'observed_rt'])
                predicted_median_go_rt = np.median(combined_go_rt)
                axes[i, 0].axvline(observed_median_go_rt, color='blue', linestyle='--', linewidth=2, label='Observed Median')
                axes[i, 0].axvline(predicted_median_go_rt, color='orange', linestyle='--', linewidth=2, label='Predicted Median')

                axes[i, 0].set_title(f"Go RT Distribution - Participant {i + 1}")
                axes[i, 0].set_xlabel("RT (ms)")
                axes[i, 0].set_ylabel("Density")
                axes[i, 0].legend()

                # Plot 2: Signal-Respond RT Density for Each Participant
                sns.histplot(participant_data.loc[participant_data['outcome'] == 'stop-respond', 'observed_rt'], bins=10, color='black', ax=axes[i, 1], kde=True, alpha=0.1, label='Observed Signal-Respond RT')
                predicted_signal_respond_rts = np.concatenate([trial_df.loc[(trial_df['trial_type'] == 'stop') & (trial_df['outcome'] == 'stop-respond'), 'observed_rt'].values for trial_df in participant_simulated_trials])
                sns.kdeplot(predicted_signal_respond_rts, ax=axes[i, 1], color='gray', alpha=0.5, label='Predicted Signal-Respond RT')

                # Calculate medians and plot as dashed lines
                observed_median_sr = np.median(participant_data.loc[participant_data['outcome'] == 'stop-respond', 'observed_rt'])
                predicted_median_sr = np.median(predicted_signal_respond_rts)
                axes[i, 1].axvline(observed_median_sr, color='blue', linestyle='--', linewidth=2, label='Observed Median')
                axes[i, 1].axvline(predicted_median_sr, color='orange', linestyle='--', linewidth=2, label='Predicted Median')

                axes[i, 1].set_title(f"Signal-Respond RT Distribution - Participant {i + 1}")
                axes[i, 1].set_xlabel("RT (ms)")
                axes[i, 1].set_ylabel("Density")
                axes[i, 1].legend()

                # Plot 3: Signal-Respond Rate by SSD
                # Extract SSDs for this specific participant
                ssds_participant = participant_data.loc[participant_data['trial_type'] == 'stop', 'ssd'].unique()
                ssds_sorted = sorted([int(round(ssd)) for ssd in ssds_participant])

                # Calculate observed response rate for each SSD for this participant
                observed_rr_participant = [
                    np.mean(participant_data[(participant_data['trial_type'] == 'stop') & (participant_data['ssd'] == ssd)]['outcome'] == 'stop-respond')
                    for ssd in ssds_sorted
                ]

                predicted_medians = []
                for ssd_idx, ssd in enumerate(ssds_sorted):
                    # Extract the predicted response rates (RR) for each SSD from all simulated trials
                    predicted_rr_for_ssd = [
                        np.mean(trial_df[(trial_df['ssd'] == ssd) & (trial_df['trial_type'] == 'stop')]['outcome'] == 'stop-respond')
                        for trial_df in participant_simulated_trials
                    ]

                    # Violin plot of the predicted response rates for each SSD at its specific position
                    sns.violinplot(
                        x=[ssd_idx] * len(predicted_rr_for_ssd),  # Positioning for each SSD
                        y=predicted_rr_for_ssd,
                        ax=axes[i, 2],
                        color="gray",
                        inner=None,
                        bw_adjust=0.2
                    )

                    # Boxplot for predicted RR within the violin plot, centered at SSD position
                    sns.boxplot(
                        x=[ssd_idx] * len(predicted_rr_for_ssd),
                        y=predicted_rr_for_ssd,
                        ax=axes[i, 2],
                        color="black",
                        width=0.15
                    )

                    # Calculate and plot the median of predicted RR for this SSD
                    median_pred_rr = np.median(predicted_rr_for_ssd)
                    predicted_medians.append(median_pred_rr)
                    axes[i, 2].plot([ssd_idx - 0.2, ssd_idx + 0.2], [median_pred_rr, median_pred_rr], 'k-', label='Predicted RR' if ssd_idx == 0 else "")

                    # Plot observed RR for this SSD as a larger marker with 'kx' style
                    axes[i, 2].plot(
                        ssd_idx,
                        observed_rr_participant[ssd_idx],
                        'kx',
                        markersize=8,
                        label="Observed RR" if ssd_idx == 0 else ""
                    )

                # Connect the medians of the predicted RRs across SSDs with a solid black line
                axes[i, 2].plot(range(len(ssds_sorted)), predicted_medians, 'k-', linewidth=2, label='Predicted Median RR')

                # Add horizontal dashed and dotted lines for the posterior median and 95% CI of P(TF)
                tf_median = np.median(p_tf_samples[:, i])
                tf_ci_lower, tf_ci_upper = np.percentile(p_tf_samples[:, i], [2.5, 97.5])
                axes[i, 2].axhline(tf_median, color='black', linestyle=':', linewidth=1, label="Posterior Median P(TF)")
                axes[i, 2].axhline(tf_ci_lower, color='black', linestyle='-.', linewidth=1, label="95% CI P(TF)")
                axes[i, 2].axhline(tf_ci_upper, color='black', linestyle='-.')

                # Set x-axis labels to SSD values for clarity
                axes[i, 2].set_xticks(range(len(ssds_sorted)))
                axes[i, 2].set_xticklabels(ssds_sorted)
                axes[i, 2].set_title(f"Inhibition Function - Participant {i + 1}")
                axes[i, 2].set_xlabel("SSD (ms)")
                axes[i, 2].set_ylabel("P(respond | stop signal)")
                axes[i, 2].legend()

            plt.tight_layout()
            plt.show()

    if cross_validation_mode:
        # Return cross-validation metrics
        return {
            'simulated_trials_primary': all_simulated_trials,
            'simulated_trials_comparison': all_simulated_trials_comp,
            'diff_go_rt_median': diff_go_rt_median,
            'diff_sr_rate': diff_sr_rate,
            'diff_sr_rt_median': diff_sr_rt_median,
            'summary': {
                'mean_diff_go_rt': np.mean(diff_go_rt_median),
                'std_diff_go_rt': np.std(diff_go_rt_median),
                'mean_diff_sr_rate': np.mean(diff_sr_rate),
                'std_diff_sr_rate': np.std(diff_sr_rate),
                'mean_diff_sr_rt': np.mean(diff_sr_rt_median) if len(diff_sr_rt_median) > 0 else None,
                'std_diff_sr_rt': np.std(diff_sr_rt_median) if len(diff_sr_rt_median) > 0 else None,
            }
        }
    else:
        # Return traditional PPC metrics
        return {
            'simulated_trials': all_simulated_trials,
            'p_values_go': p_values_go,
            'p_values_sr': p_values_sr,
            'p_values_sr_rt': p_values_sr_rt,
        }