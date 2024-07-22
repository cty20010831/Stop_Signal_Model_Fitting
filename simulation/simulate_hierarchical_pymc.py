import os
import pymc as pm
import pandas as pd
import argparse

from util import simulate_trials_fixed_SSD, simulate_trials_staircase_SSD

# Set random seed for reproducibility
SEED = 42

# Constants for the simulation
N = 100  # Number of participants
T = 50  # Total number of trials per participant
TRIAL_TYPE_SEQUENCE = ["go", "go", "go", "go", "stop"] * 10
FIXED_SSD_SET = [80, 160, 240, 320, 400, 480]
STARTING_STAIRCASE_SSD = 200

def main():
    # Get the directory where this script is located at
    dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(description='Simulate SSD trial data using hierarchical Bayesian models.')
    parser.add_argument('--type', type=str, choices=['fixed', 'staircase'], required=True,
                        help='Type of SSD simulation to perform: "fixed" or "staircase".')

    # Parse command-line arguments
    args = parser.parse_args()

    # Sample (individual-level) parameters from hierarchical prior distribution
    with pm.Model():
        # Group-level parameters with priors
        mu_mu_go = pm.TruncatedNormal('mu_mu_go', mu=500, sigma=50, lower=0.001, upper=1000, initval=500)
        # sigma_mu_go = pm.Uniform('sigma_mu_go', lower=1, upper=500, initval=50)
        sigma_mu_go = pm.Gamma('sigma_mu_go', alpha=8, beta=0.1, initval=80)
        mu_sigma_go = pm.TruncatedNormal('mu_sigma_go', mu=150, sigma=50, lower=1, upper=500, initval=150)
        # sigma_sigma_go = pm.Uniform('sigma_sigma_go', lower=1, upper=500, initval=50)
        sigma_sigma_go = pm.Gamma('sigma_sigma_go', alpha=8, beta=0.1, initval=80)
        mu_tau_go = pm.TruncatedNormal('mu_tau_go', mu=150, sigma=50, lower=1, upper=500, initval=150)
        # sigma_tau_go = pm.Uniform('sigma_tau_go', lower=1, upper=500, initval=50)
        sigma_tau_go = pm.Gamma('sigma_tau_go', alpha=8, beta=0.1, initval=80)

        mu_mu_stop = pm.TruncatedNormal('mu_mu_stop', mu=300, sigma=50, lower=0.001, upper=1000, initval=300)
        # sigma_mu_stop = pm.Uniform('sigma_mu_stop', lower=1, upper=500, initval=50)
        sigma_mu_stop = pm.Gamma('sigma_mu_stop', alpha=8, beta=0.1, initval=80)
        mu_sigma_stop = pm.TruncatedNormal('mu_sigma_stop', mu=150, sigma=50, lower=1, upper=500, initval=150)
        # sigma_sigma_stop = pm.Uniform('sigma_sigma_stop', lower=1, upper=500, initval=50)
        sigma_sigma_stop = pm.Gamma('sigma_sigma_stop', alpha=8, beta=0.1, initval=80)
        mu_tau_stop = pm.TruncatedNormal('mu_tau_stop', mu=150, sigma=50, lower=1, upper=500, initval=150)
        # sigma_tau_stop = pm.Uniform('sigma_tau_stop', lower=1, upper=500, initval=50)
        sigma_tau_stop = pm.Gamma('sigma_tau_stop', alpha=8, beta=0.1, initval=80)

        # mu_p_tf = pm.TruncatedNormal('mu_p_tf', mu=0.05, sigma=1, lower=-1.5, upper=1.5, initval=0.05)
        mu_p_tf = pm.TruncatedNormal('mu_p_tf', mu=0.05, sigma=0.1, lower=-1.5, upper=1.5, initval=0.05)
        # sigma_p_tf = pm.Uniform('sigma_p_tf', lower=0.01, upper=3, initval=0.1)
        sigma_p_tf = pm.Gamma('sigma_p_tf', alpha=2, beta=1, initval=2)

        # Participant-specific parameters
        mu_go = pm.TruncatedNormal('mu_go', mu=mu_mu_go, sigma=sigma_mu_go, lower=0.001, upper=1000)
        sigma_go = pm.TruncatedNormal('sigma_go', mu=mu_sigma_go, sigma=sigma_sigma_go, lower=1, upper=500)
        tau_go = pm.TruncatedNormal('tau_go', mu=mu_tau_go, sigma=sigma_tau_go, lower=1, upper=500)
        mu_stop = pm.TruncatedNormal('mu_stop', mu=mu_mu_stop, sigma=sigma_mu_stop, lower=0.001, upper=1000)
        sigma_stop = pm.TruncatedNormal('sigma_stop', mu=mu_sigma_stop, sigma=sigma_sigma_stop, lower=1, upper=500)
        tau_stop = pm.TruncatedNormal('tau_stop', mu=mu_tau_stop, sigma=sigma_tau_stop, lower=1, upper=500)

        p_tf_probit = pm.TruncatedNormal('p_tf_probit', mu=mu_p_tf, sigma=sigma_p_tf, lower=-1.5, upper=1.5)
        p_tf = pm.Deterministic('p_tf', pm.math.invprobit(p_tf_probit))
    
    # Draw N samples from the prior distribution
    draws = pm.draw([mu_go, sigma_go, tau_go, mu_stop, sigma_stop, tau_stop, p_tf], 
                    draws=N, random_seed=SEED)
    
    # Initialize a DataFrame to store true parameters
    true_parameters = pd.DataFrame(columns=[
        'participant_id', 'mu_go', 'sigma_go', 'tau_go', 
        'mu_stop', 'sigma_stop', 'tau_stop', 'p_tf'])

    # Conditional execution based on input
    all_trials = []
    for i in range(N):
        participant_id = i

        # Retrieve individual parameters
        mu_go = draws[0][i]
        sigma_go = draws[1][i]
        tau_go = draws[2][i]
        mu_stop = draws[3][i]
        sigma_stop = draws[4][i]
        tau_stop = draws[5][i]
        p_tf = draws[6][i]

        # Store the true parameters
        true_parameters = pd.concat([true_parameters, pd.DataFrame({
            'participant_id': [participant_id],
            'mu_go': [mu_go],
            'sigma_go': [sigma_go],
            'tau_go': [tau_go],
            'mu_stop': [mu_stop],
            'sigma_stop': [sigma_stop],
            'tau_stop': [tau_stop],
            'p_tf': [p_tf]
        })], ignore_index=True)
        
        if args.type == 'fixed':
            # Simulate fixed SSD dataset
            trial_df = simulate_trials_fixed_SSD(
                TRIAL_TYPE_SEQUENCE, FIXED_SSD_SET, p_tf, 
                mu_go, sigma_go, tau_go, mu_stop, sigma_stop, tau_stop
            )
        elif args.type == 'staircase':
            # Simulate staircase SSD dataset
            trial_df = simulate_trials_staircase_SSD(
                TRIAL_TYPE_SEQUENCE, STARTING_STAIRCASE_SSD, p_tf,
                mu_go, sigma_go, tau_go, mu_stop, sigma_stop, tau_stop
            )
        
        trial_df['participant_id'] = participant_id
        all_trials.append(trial_df)

    # Save simulated data and true parameters
    simulated_data = pd.concat(all_trials, ignore_index=True)
    simulated_data_file_name = os.path.join(dir, f"simulated_data/hierarchical_simulated_data_{args.type}_SSD.csv")
    simulated_data.to_csv(simulated_data_file_name, index=False)
    print(f"Saved simulated data ({args.type} SSD) for {N} participants with {T} trials each.")

    true_parameters_file_name = os.path.join(dir, f"true_param/hierarchical_true_parameters_{args.type}_SSD.csv")
    true_parameters.to_csv(true_parameters_file_name, index=False)
    print(f"Saved true parameters ({args.type} SSD) for {N} participants with {T} trials each.")


if __name__ == '__main__':
    main()

# Example usage:
# python simulation/simulate_hierarchical_pymc.py --type fixed
# python simulation/simulate_hierarchical_pymc.py --type staircase