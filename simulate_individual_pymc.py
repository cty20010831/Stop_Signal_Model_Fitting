import pymc as pm
import pandas as pd
import argparse

from util import simulate_trials_fixed_SSD, simulate_trials_staircase_SSD

# Set random seed
SEED = 42

# Define some constants for this simulation
N = 100  # number of participants
T = 50  # total number of trials
trial_type_sequence = ["go", "go", "go", "go", "stop"] * 10
fixed_ssd_set = [80, 160, 240, 320, 400, 480]
starting_staircase_ssd = 200

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Simulate trial data for fixed or staircase SSD.')
    parser.add_argument('--type', type=str, choices=['fixed', 'staircase'], required=True,
                        help='Specify the type of data to generate: "fixed" for fixed SSD or "staircase" for staircase SSD.')
    args = parser.parse_args()

    # Sample parameters based on prior distribution
    with pm.Model():
        # Define the distributions for the parameters using PyMC
        mu_go = pm.Uniform('mu_go', lower=1, upper=1000)
        sigma_go = pm.Uniform('sigma_go', lower=1, upper=300)
        tau_go = pm.Uniform('tau_go', lower=1, upper=300)
        mu_stop = pm.Uniform('mu_stop', lower=1, upper=600)
        sigma_stop = pm.Uniform('sigma_stop', lower=1, upper=250)
        tau_stop = pm.Uniform('tau_stop', lower=1, upper=250)
        p_tf = pm.Uniform('p_tf', lower=0.0, upper=1.0)

    # Draw N samples from the prior distribution
    draws = pm.draw([mu_go, sigma_go, tau_go, mu_stop, sigma_stop, tau_stop, p_tf], 
                    draws=N, random_seed=SEED)
    
    all_trials = []

    # Run simulation based on the type of SSD specified
    for i in range(N):
        participant_id = i + 1  # Participant identifier

        # Retrieve individual parameters
        mu_go = draws[0][i]
        sigma_go = draws[1][i]
        tau_go = draws[2][i]
        mu_stop = draws[3][i]
        sigma_stop = draws[4][i]
        tau_stop = draws[5][i]
        p_tf = draws[6][i]
        
        if args.type == 'fixed':
            # Simulate fixed SSD dataset
            trial_df = simulate_trials_fixed_SSD(
                trial_type_sequence, fixed_ssd_set, p_tf, 
                mu_go, sigma_go, tau_go, mu_stop, sigma_stop, tau_stop
            )
        elif args.type == 'staircase':
            # Simulate staircase SSD dataset
            trial_df = simulate_trials_staircase_SSD(
                trial_type_sequence, starting_staircase_ssd, p_tf,
                mu_go, sigma_go, tau_go, mu_stop, sigma_stop, tau_stop
            )
    
        trial_df['participant_id'] = participant_id
        all_trials.append(trial_df)

    simulated_data = pd.concat(all_trials, ignore_index=True)
    file_name = f"simulated_data/individual_simulated_data_{args.type}_SSD.csv"
    simulated_data.to_csv(file_name, index=False)
    print(f"Saved simulated data ({args.type} SSD) for {N} participants with {T} trials each.")

if __name__ == '__main__':
    main()

# Example usage:
# python simulate_individual_pymc.py --type fixed
# python simulate_individual_pymc.py --type staircase