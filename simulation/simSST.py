"""
This python script simulates data from a staircase stop-signal task (SST)
with trigger failures, based on the logic from Matzke et al. (2016).

Please refer to simulation_pseudo_code.txt for detailed explanation of the logic. 

Another important thing is that this simulation logic matches the experimental 
settings from the jsPsych version of STOP-IT experiment 
(https://github.com/fredvbrug/STOP-IT/tree/master/jsPsych_version). It contains 
25% of stop trials (trial sequence randomly assigned). In terms of the starting 
SSDs, they are set to be a fixed value of 200 in the jsPsych version. But, the 
data I saw right now seems to be random sampling from a potential set of values.
Will need to confirm. 

There is also an accompanied R shiny package:
shiny::runGitHub('STOP-IT', 'fredvbrug', launch.browser = TRUE, subdir = 'jsPsych_version')
It outputs many (11 in total) dependent variables (DVs) that may be useful for ppc.

To have example usage, please run this script directly:
python3 simulation/simSST.py
"""

import pandas as pd
import random
from scipy.stats import exponnorm

"""
Staircase Stop Signal Task (SST) Simulation
Based on the simplified trigger failure logic from Matzke et al. (2016)
"""


def simulate_exgaussian(mu, sigma, tau):
    """
    Generate a random value following Ex-Gaussian distribution.

    Parameters
    ----------
    mu : float
        Mean of the Gaussian component.
    sigma : float
        Standard deviation of the Gaussian component.
    tau : float
        Mean (or rate parameter) of the exponential component.

    Returns
    -------
    float
        A simulated value following Ex-Gaussian distribution.
    """
    K = tau / sigma

    while True:
        simulated_value = exponnorm.rvs(K=K, loc=mu, scale=sigma, size=1)[0]
        if simulated_value >= 0:
            return simulated_value


def simulate_stop_signal_trial(go_rt, ssd, mu_stop, sigma_stop, tau_stop, p_tf):
    """
    Simulate a single stop-signal trial with trigger failures.

    This implements the simplified trigger failure logic where:
    - With probability p_tf, the stop process never starts (trigger failure)
    - Otherwise, a normal race between go and stop processes occurs

    Parameters
    ----------
    go_rt : float
        Pre-generated go reaction time for this trial.
    ssd : float
        Stop-signal delay for this trial.
    mu_stop : float
        Mean of the Gaussian component for SSRT.
    sigma_stop : float
        Standard deviation of the Gaussian component for SSRT.
    tau_stop : float
        Mean of the exponential component for SSRT.
    p_tf : float
        Probability of trigger failure (0 to 1).

    Returns
    -------
    dict
        Dictionary containing:
        - 'outcome': 'stop-respond' or 'successful_inhibition'
        - 'response_rt': observed RT (None for successful inhibition)
        - 'ss_rt': stop-signal reaction time (unobserved)
        - 'ssd': stop-signal delay used
    """
    # Step 1: Check if trigger fails
    if random.random() < p_tf:
        # Trigger failure - stop process never started
        return {
            'outcome': 'stop-respond',
            'response_rt': go_rt,
            'ss_rt': None,  # Stop process never initiated
            'ssd': ssd
        }

    # Step 2: Normal race model - stop process was triggered
    ss_rt = simulate_exgaussian(mu_stop, sigma_stop, tau_stop)

    # Determine race outcome
    if go_rt < (ssd + ss_rt):
        # Go process wins - response executed
        return {
            'outcome': 'stop-respond',
            'response_rt': go_rt,
            'ss_rt': ss_rt,
            'ssd': ssd
        }
    else:
        # Stop process wins - response inhibited
        return {
            'outcome': 'successful_inhibition',
            'response_rt': None,
            'ss_rt': ss_rt,
            'ssd': ssd
        }


def update_ssd(current_ssd, outcome, step_size, min_ssd, max_ssd):
    """
    Adjust SSD based on previous stop trial outcome (staircase procedure).

    Parameters
    ----------
    current_ssd : float
        Current stop-signal delay.
    outcome : str
        Outcome of the previous stop trial ('successful_inhibition' or 'stop-respond').
    step_size : float
        Amount to increase/decrease SSD.
    min_ssd : float, optional
        Minimum allowed SSD (default: 50).
    max_ssd : float, optional
        Maximum allowed SSD (default: 1000).

    Returns
    -------
    float
        Updated SSD for the next stop trial.
    """
    if outcome == 'successful_inhibition':
        # Make it harder - increase SSD
        new_ssd = current_ssd + step_size
    else:  # stop-respond (failed inhibition)
        # Make it easier - decrease SSD
        new_ssd = current_ssd - step_size

    # Apply bounds
    new_ssd = max(min_ssd, min(new_ssd, max_ssd))

    return new_ssd


def simulate_staircase_block(n_trials, prop_stop,
                             mu_go, sigma_go, tau_go,
                             mu_stop, sigma_stop, tau_stop,
                             p_tf,
                             ssd_start=None, step_size=50,
                             min_ssd=50, max_ssd=1000):
    """
    Simulate a complete block of stop-signal trials with staircase SSD adjustment.

    Parameters
    ----------
    n_trials : int
        Total number of trials in the block.
    prop_stop : float
        Proportion of stop trials (e.g., 0.25 for 25% stop trials).
    mu_go : float
        Mean of the Gaussian component for go RT.
    sigma_go : float
        Standard deviation of the Gaussian component for go RT.
    tau_go : float
        Mean of the exponential component for go RT.
    mu_stop : float
        Mean of the Gaussian component for SSRT.
    sigma_stop : float
        Standard deviation of the Gaussian component for SSRT.
    tau_stop : float
        Mean of the exponential component for SSRT.
    p_tf : float
        Probability of trigger failure (0 to 1).
    ssd_start : float or None, optional
        Starting SSD. If None, randomly picks from [150, 200, 250, 300].
    step_size : float, optional
        Staircase increment/decrement (default: 50 ms).
    min_ssd : float, optional
        Minimum allowed SSD (default: 50).
    max_ssd : float, optional
        Maximum allowed SSD (default: 1000).

    Returns
    -------
    pd.DataFrame
        DataFrame containing all trials with columns:
        - 'trial_type': 'go' or 'stop'
        - 'ssd': stop-signal delay (None for go trials)
        - 'observed_rt': observed RT (None for successful inhibitions)
        - 'ss_rt': stop-signal RT (None for go trials and trigger failures)
        - 'outcome': 'go', 'stop-respond', or 'successful_inhibition'
        - 'trial_number': sequential trial number (additional)
        - 'go_rt': generated go RT for this trial (additional)
    """
    # Calculate number of go and stop trials from total trials and proportion
    n_stop = int(n_trials * prop_stop)
    n_go = n_trials - n_stop

    # Initialize starting SSD
    if ssd_start is None:
        ssd_start = random.choice([150, 200, 250, 300])

    current_ssd = ssd_start
    trials = []

    # Create randomized trial sequence
    trial_types = ['go'] * n_go + ['stop'] * n_stop
    random.shuffle(trial_types)

    # Simulate each trial
    for trial_num, trial_type in enumerate(trial_types, start=1):
        # Generate go RT for this trial
        go_rt = simulate_exgaussian(mu_go, sigma_go, tau_go)

        if trial_type == 'go':
            # Go trial - no stop signal
            trials.append({
                'trial_number': trial_num,
                'trial_type': 'go',
                'go_rt': go_rt,
                'ssd': None,
                'ss_rt': None,
                'outcome': 'go',
                'observed_rt': go_rt
            })

        else:  # trial_type == 'stop'
            # Stop trial with potential trigger failure
            result = simulate_stop_signal_trial(
                go_rt, current_ssd,
                mu_stop, sigma_stop, tau_stop,
                p_tf
            )

            trials.append({
                'trial_number': trial_num,
                'trial_type': 'stop',
                'go_rt': go_rt,
                'ssd': result['ssd'],
                'ss_rt': result['ss_rt'],
                'outcome': result['outcome'],
                'observed_rt': result['response_rt']
            })

            # Update SSD for next stop trial
            current_ssd = update_ssd(
                current_ssd,
                result['outcome'],
                step_size,
                min_ssd,
                max_ssd
            )

    return pd.DataFrame(trials)


def simulate_participant_blocks(n_blocks, n_trials_per_block, prop_stop=0.25,
                                mu_go=None, sigma_go=None, tau_go=None,
                                mu_stop=None, sigma_stop=None, tau_stop=None,
                                p_tf=None,
                                ssd_start=None, step_size=50,
                                min_ssd=50, max_ssd=1000,
                                include_block_id=True):
    """
    Simulate multiple blocks for a single participant with continuous SSD tracking.

    The SSD is updated across all trials and carries over between blocks
    (not reset at the start of each block). This matches the actual experiment
    behavior where SSD continues to adapt across the entire session.

    Parameters
    ----------
    n_blocks : int
        Number of blocks to simulate.
    n_trials_per_block : int
        Total number of trials per block.
    prop_stop : float, optional
        Proportion of stop trials (default: 0.25 for 25% stop trials).
    mu_go, sigma_go, tau_go : float
        Go process Ex-Gaussian parameters.
    mu_stop, sigma_stop, tau_stop : float
        Stop process Ex-Gaussian parameters.
    p_tf : float
        Probability of trigger failure.
    ssd_start : float or None, optional
        Starting SSD. If None, randomly picks from [150, 200, 250, 300].
    step_size : float, optional
        Staircase increment/decrement in ms. Default: 50.
    min_ssd : float, optional
        Minimum allowed SSD. Default: 50.
    max_ssd : float, optional
        Maximum allowed SSD. Default: 1000.
    include_block_id : bool, optional
        Whether to include 'block_id' column. Default: True.

    Returns
    -------
    pd.DataFrame
        Combined data from all blocks with continuous trial numbering and SSD tracking.
    """
    # Calculate number of go and stop trials per block
    n_stop_per_block = int(n_trials_per_block * prop_stop)
    n_go_per_block = n_trials_per_block - n_stop_per_block

    # Initialize starting SSD
    if ssd_start is None:
        ssd_start = random.choice([150, 200, 250, 300])

    current_ssd = ssd_start
    all_trials = []
    global_trial_num = 1

    for block_id in range(n_blocks):
        # Create randomized trial sequence for this block
        trial_types = ['go'] * n_go_per_block + ['stop'] * n_stop_per_block
        random.shuffle(trial_types)

        # Simulate each trial in this block
        for trial_type in trial_types:
            # Generate go RT for this trial
            go_rt = simulate_exgaussian(mu_go, sigma_go, tau_go)

            if trial_type == 'go':
                # Go trial - no stop signal
                trial_data = {
                    'trial_number': global_trial_num,
                    'trial_type': 'go',
                    'go_rt': go_rt,
                    'ssd': None,
                    'ss_rt': None,
                    'outcome': 'go',
                    'observed_rt': go_rt
                }
                if include_block_id:
                    trial_data['block_id'] = block_id

                all_trials.append(trial_data)

            else:  # trial_type == 'stop'
                # Stop trial with potential trigger failure
                result = simulate_stop_signal_trial(
                    go_rt, current_ssd,
                    mu_stop, sigma_stop, tau_stop,
                    p_tf
                )

                trial_data = {
                    'trial_number': global_trial_num,
                    'trial_type': 'stop',
                    'go_rt': go_rt,
                    'ssd': result['ssd'],
                    'ss_rt': result['ss_rt'],
                    'outcome': result['outcome'],
                    'observed_rt': result['response_rt']
                }
                if include_block_id:
                    trial_data['block_id'] = block_id

                all_trials.append(trial_data)

                # Update SSD for next stop trial (carries across blocks!)
                current_ssd = update_ssd(
                    current_ssd,
                    result['outcome'],
                    step_size,
                    min_ssd,
                    max_ssd
                )

            global_trial_num += 1

    return pd.DataFrame(all_trials)


def simulate_experiment(n_participants, n_blocks, n_trials_per_block, prop_stop=0.25,
                        participant_params=None,
                        ssd_start=None, step_size=50,
                        min_ssd=50, max_ssd=1000,
                        include_block_id=True):
    """
    General-purpose simulation function for multiple participants and multiple blocks.

    This is the main function for generating complete experimental datasets with
    staircase stop-signal tasks. SSD is tracked continuously across all blocks
    for each participant (not reset between blocks), matching actual experiment behavior.

    Parameters
    ----------
    n_participants : int
        Number of participants to simulate.
    n_blocks : int
        Number of blocks per participant.
    n_trials_per_block : int
        Total number of trials per block.
    prop_stop : float, optional
        Proportion of stop trials (default: 0.25 for 25% stop trials).
    participant_params : pd.DataFrame or dict
        Parameters for each participant. Must have columns/keys:
        ['mu_go', 'sigma_go', 'tau_go', 'mu_stop', 'sigma_stop', 'tau_stop', 'p_tf']
        with n_participants rows/entries.
    ssd_start : float or None, optional
        Starting SSD for the first block. If None, randomly picks from [150, 200, 250, 300]
        for each participant. Default: None.
    step_size : float, optional
        Staircase increment/decrement in ms. Default: 50.
    min_ssd : float, optional
        Minimum allowed SSD in ms. Default: 50.
    max_ssd : float, optional
        Maximum allowed SSD in ms. Default: 1000.
    include_block_id : bool, optional
        Whether to include 'block_id' column in output. Default: True.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with columns:
        - 'trial_type': 'go' or 'stop'
        - 'ssd': stop-signal delay (None for go trials)
        - 'observed_rt': observed RT (None for successful inhibitions)
        - 'ss_rt': stop-signal RT (None for go trials and trigger failures)
        - 'outcome': 'go', 'stop-respond', or 'successful_inhibition'
        - 'participant_id': participant identifier (0 to n_participants-1)
        - 'block_id': block identifier (0 to n_blocks-1) [if include_block_id=True]
        - Additional columns: 'trial_number', 'go_rt'

    Examples
    --------
    >>> # Simulate 10 participants, 2 blocks each, 150 trials per block (25% stop trials)
    >>> import pandas as pd
    >>> params = pd.DataFrame({
    ...     'mu_go': [450]*10,
    ...     'sigma_go': [60]*10,
    ...     'tau_go': [120]*10,
    ...     'mu_stop': [220]*10,
    ...     'sigma_stop': [35]*10,
    ...     'tau_stop': [60]*10,
    ...     'p_tf': [0.1]*10
    ... })
    >>> data = simulate_experiment(
    ...     n_participants=10,
    ...     n_blocks=2,
    ...     n_trials_per_block=150,
    ...     prop_stop=0.25,
    ...     participant_params=params
    ... )
    """
    all_data = []

    for participant_id in range(n_participants):
        # Extract parameters for this participant
        if isinstance(participant_params, pd.DataFrame):
            params = participant_params.iloc[participant_id]
        else:
            params = {k: v[participant_id] for k, v in participant_params.items()}

        # Simulate all blocks for this participant with continuous SSD tracking
        participant_data = simulate_participant_blocks(
            n_blocks=n_blocks,
            n_trials_per_block=n_trials_per_block,
            prop_stop=prop_stop,
            mu_go=params['mu_go'],
            sigma_go=params['sigma_go'],
            tau_go=params['tau_go'],
            mu_stop=params['mu_stop'],
            sigma_stop=params['sigma_stop'],
            tau_stop=params['tau_stop'],
            p_tf=params['p_tf'],
            ssd_start=ssd_start,
            step_size=step_size,
            min_ssd=min_ssd,
            max_ssd=max_ssd,
            include_block_id=include_block_id
        )

        # Add participant identifier
        participant_data['participant_id'] = participant_id
        all_data.append(participant_data)

    return pd.concat(all_data, ignore_index=True)

def example_usage():
    # Example parameters for 3 participants
    participant_params = pd.DataFrame({
        'mu_go': [450, 500, 480],
        'sigma_go': [60, 70, 65],
        'tau_go': [120, 130, 125],
        'mu_stop': [220, 240, 230],
        'sigma_stop': [35, 40, 38],
        'tau_stop': [60, 70, 65],
        'p_tf': [0.1, 0.15, 0.12]
    })

    # Simulate experiment with 150 trials per block (25% stop trials = ~37 stop, ~113 go)
    data = simulate_experiment(
        n_participants=3,
        n_blocks=2,
        n_trials_per_block=150,
        prop_stop=0.25,
        participant_params=participant_params
    )

    print(pd.concat([data.head(30), data.tail(30)]))

if __name__ == "__main__":
    example_usage()