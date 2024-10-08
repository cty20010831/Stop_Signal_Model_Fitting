import os
import pymc as pm
import pandas as pd
import argparse

from util import simulate_trials_fixed_SSD, simulate_trials_staircase_SSD

# Set random seed for reproducibility
SEED = 42
FIXED_SSD_SET = [80, 160, 240, 320, 400, 480]
STARTING_STAIRCASE_SSD = 200
TRIAL_TYPE_SEQUENCE_BASE = ["go", "stop", "stop", "stop", "stop"]

def main():
    # Get the directory where this script is located at
    dir = os.path.dirname(os.path.abspath(__file__))

    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Simulate trial data for fixed or staircase SSD.')
    parser.add_argument('--type', type=str, choices=['fixed', 'staircase'], required=True,
                        help='Specify the type of data to generate: "fixed" for fixed SSD or "staircase" for staircase SSD.')
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

    # Sample parameters based on prior distribution
    with pm.Model():
        # Define the distributions for the parameters using PyMC
        mu_go = pm.Uniform('mu_go', lower=0.001, upper=1000, initval=500)
        sigma_go = pm.Uniform('sigma_go', lower=1, upper=500, initval=50)
        tau_go = pm.Uniform('tau_go', lower=1, upper=500, initval=50)
        mu_stop = pm.Uniform('mu_stop', lower=0.001, upper=1000, initval=300)
        sigma_stop = pm.Uniform('sigma_stop', lower=1, upper=500, initval=50)
        tau_stop = pm.Uniform('tau_stop', lower=1, upper=500, initval=50)
        p_tf = pm.Uniform('p_tf', lower=0.0, upper=1.0, initval=0.5)

    # Draw N samples from the prior distribution
    draws = pm.draw([mu_go, sigma_go, tau_go, mu_stop, sigma_stop, tau_stop, p_tf], 
                    draws=N, random_seed=SEED)
    
    # Initialize a DataFrame to store true parameters
    true_parameters = pd.DataFrame(columns=[
        'participant_id', 'mu_go', 'sigma_go', 'tau_go', 
        'mu_stop', 'sigma_stop', 'tau_stop', 'p_tf'])

    all_trials = []

    # Run simulation based on the type of SSD specified
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

    # Ensure the directories for saving data exist
    os.makedirs(os.path.join(dir, 'true_param'), exist_ok=True)
    os.makedirs(os.path.join(dir, 'simulated_data'), exist_ok=True)

    # Save simulated data and true parameters
    simulated_data = pd.concat(all_trials, ignore_index=True)
    simulated_data_file_name = os.path.join(dir, f"simulated_data/individual_simulated_data_{args.type}_SSD.csv")
    simulated_data.to_csv(simulated_data_file_name, index=False)
    print(f"Saved simulated data ({args.type} SSD) for {N} participants with {T} trials each.")

    true_parameters_file_name = os.path.join(dir, f"true_param/individual_true_parameters_{args.type}_SSD.csv")
    true_parameters.to_csv(true_parameters_file_name, index=False)
    print(f"Saved true parameters ({args.type} SSD) for {N} participants with {T} trials each.")

if __name__ == '__main__':
    main()

# Example usage:
# python simulation/simulate_individual_pymc.py --type fixed
# python simulation/simulate_individual_pymc.py --type staircase