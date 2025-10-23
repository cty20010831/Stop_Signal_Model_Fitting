# TO-DO: Just have one subject-level helper function that takes in parameters and simulates data accordingly.
# Note: our simulation is for staircase (adpative) SSD using the independet horce race model 
# (note: there are non-independent models as well, but that's not the focus here) 
import os
import pymc as pm
import pandas as pd
import argparse

from util import simulate_trials_fixed_SSD, simulate_trials_staircase_SSD
# from util import simulate_trials_fixed_SSD_no_p_tf

# Set random seed for reproducibility
SEED = 42
FIXED_SSD_SET = [80, 160, 240, 320, 400, 480]
STARTING_STAIRCASE_SSD = 200
TRIAL_TYPE_SEQUENCE_BASE = ["go", "stop", "stop", "stop", "stop"]

def main():
    # Get the directory where this script is located at
    dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(description='Simulate SSD trial data using hierarchical Bayesian models.')

    parser.add_argument('--type', type=str, choices=['fixed', 'staircase'], required=True,
                        help='Type of SSD simulation to perform: "fixed" or "staircase".')
    
    parser.add_argument('--N', type=int, default=100,
                        help='Number of participants.')
    
    parser.add_argument('--T', type=int, default=250,
                        help='Total number of trials per participant.')

    # Parse command-line arguments
    args = parser.parse_args()
    
    # Constants for the simulation
    N, T = args.N, args.T
    # TRIAL_TYPE_SEQUENCE = TRIAL_TYPE_SEQUENCE_BASE * int(T / 5)
    TRIAL_TYPE_SEQUENCE = TRIAL_TYPE_SEQUENCE_BASE * int(T / 5)

    # Sample (individual-level) parameters from hierarchical prior distribution
    with pm.Model():
                # Group-level parameters with priors
        mu_mu_go = pm.Uniform('mu_mu_go', lower=0.001, upper=1000)
        sigma_mu_go = pm.Uniform('sigma_mu_go', lower=1, upper=500)
        mu_sigma_go = pm.Uniform('mu_sigma_go', lower=1, upper=500)
        sigma_sigma_go = pm.Uniform('sigma_sigma_go', lower=1, upper=500)
        mu_tau_go = pm.Uniform('mu_tau_go', lower=1, upper=500)
        sigma_tau_go = pm.Uniform('sigma_tau_go', lower=1, upper=500)

        mu_mu_stop = pm.Uniform('mu_mu_stop', lower=0.001, upper=1000)
        sigma_mu_stop = pm.Uniform('sigma_mu_stop', lower=1, upper=500)
        mu_sigma_stop = pm.Uniform('mu_sigma_stop', lower=1, upper=500)
        sigma_sigma_stop = pm.Uniform('sigma_sigma_stop', lower=1, upper=500)
        mu_tau_stop = pm.Uniform('mu_tau_stop', lower=1, upper=500)
        sigma_tau_stop = pm.Uniform('sigma_tau_stop', lower=1, upper=500)

        mu_p_tf_probit = pm.TruncatedNormal('mu_p_tf_probit', mu=0, sigma=1, lower=-6, upper=6)
        sigma_p_tf_probit = pm.Uniform('sigma_p_tf_probit', lower=0.01, upper=3)
        
        # Participant-specific parameters
        mu_go = pm.TruncatedNormal('mu_go', mu=mu_mu_go, sigma=sigma_mu_go, lower=0.001, upper=1000)
        sigma_go = pm.TruncatedNormal('sigma_go', mu=mu_sigma_go, sigma=sigma_sigma_go, lower=1, upper=500)
        tau_go = pm.TruncatedNormal('tau_go', mu=mu_tau_go, sigma=sigma_tau_go, lower=1, upper=500)

        mu_stop = pm.TruncatedNormal('mu_stop', mu=mu_mu_stop, sigma=sigma_mu_stop, lower=0.001, upper=1000)
        sigma_stop = pm.TruncatedNormal('sigma_stop', mu=mu_sigma_stop, sigma=sigma_sigma_stop, lower=1, upper=500)
        tau_stop = pm.TruncatedNormal('tau_stop', mu=mu_tau_stop, sigma=sigma_tau_stop, lower=1, upper=500)

        p_tf_probit = pm.TruncatedNormal('p_tf_probit', mu=mu_p_tf_probit, sigma=sigma_p_tf_probit, lower=-6, upper=6)
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
            # Simulate fixed SSD dataset with no p_tf
            # trial_df = simulate_trials_fixed_SSD_no_p_tf(
            #     TRIAL_TYPE_SEQUENCE, FIXED_SSD_SET,
            #     mu_go, sigma_go, tau_go, mu_stop, sigma_stop, tau_stop
            # )
        elif args.type == 'staircase':
            # Simulate staircase SSD dataset
            trial_df = simulate_trials_staircase_SSD(
                TRIAL_TYPE_SEQUENCE, STARTING_STAIRCASE_SSD, p_tf,
                mu_go, sigma_go, tau_go, mu_stop, sigma_stop, tau_stop
            )
        
        trial_df['participant_id'] = participant_id
        all_trials.append(trial_df)

    # Ensure the directories for saving data exist
    os.makedirs(os.path.join(dir, 'true_param'), exist_ok=True)
    os.makedirs(os.path.join(dir, 'simulated_data'), exist_ok=True)

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