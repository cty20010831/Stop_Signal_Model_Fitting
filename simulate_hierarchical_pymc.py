import pymc as pm
import pandas as pd
import argparse

from util import simulate_trials_fixed_SSD, simulate_trials_staircase_SSD

# Set random seed for reproducibility
SEED = 42

# Constants for the simulation
N = 100  # Number of participants
T = 50  # Total number of trials per participant
trial_type_sequence = ["go", "go", "go", "go", "stop"] * 10
fixed_ssd_set = [80, 160, 240, 320, 400, 480]
starting_staircase_ssd = 200

def main():
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(description='Simulate SSD trial data using hierarchical Bayesian models.')
    parser.add_argument('--type', type=str, choices=['fixed', 'staircase'], required=True,
                        help='Type of SSD simulation to perform: "fixed" or "staircase".')

    # Parse command-line arguments
    args = parser.parse_args()

    # Sample (individual-level) parameters from hierarchical prior distribution
    with pm.Model():
        # Group-level parameters
        mu_mu_go = pm.TruncatedNormal('mu_mu_go', mu=500, sigma=0.0001, lower=0)
        sigma_mu_go = pm.Uniform('sigma_mu_go', lower=0, upper=300)
        mu_sigma_go = pm.TruncatedNormal('mu_sigma_go', mu=100, sigma=0.0001, lower=0)
        sigma_sigma_go = pm.Uniform('sigma_sigma_go', lower=0, upper=200)
        mu_tau_go = pm.TruncatedNormal('mu_tau_go', mu=80, sigma=0.0001, lower=0)
        sigma_tau_go = pm.Uniform('sigma_tau_go', lower=0, upper=200)

        mu_mu_stop = pm.TruncatedNormal('mu_mu_stop', mu=200, sigma=0.0001, lower=0)
        sigma_mu_stop = pm.Uniform('sigma_mu_stop', lower=0, upper=200)
        mu_sigma_stop = pm.TruncatedNormal('mu_sigma_stop', mu=40, sigma=0.0001, lower=0)
        sigma_sigma_stop = pm.Uniform('sigma_sigma_stop', lower=0, upper=100)
        mu_tau_stop = pm.TruncatedNormal('mu_tau_stop', mu=30, sigma=0.0001, lower=0)
        sigma_tau_stop = pm.Uniform('sigma_tau_stop', lower=0, upper=100)
        
        mu_p_tf = pm.TruncatedNormal('mu_p_tf', mu=0, sigma=1, lower=-6, upper=6)
        sigma_p_tf = pm.Uniform('sigma_p_tf', lower=0.01, upper=3)

        # Participant-specific parameters
        mu_go = pm.TruncatedNormal('mu_go', mu=mu_mu_go, sigma=sigma_mu_go, lower=0)
        sigma_go = pm.TruncatedNormal('sigma_go', mu=mu_sigma_go, sigma=sigma_sigma_go, lower=1)
        tau_go = pm.TruncatedNormal('tau_go', mu=mu_tau_go, sigma=sigma_tau_go, lower=1)
        mu_stop = pm.TruncatedNormal('mu_stop', mu=mu_mu_stop, sigma=sigma_mu_stop, lower=0)
        sigma_stop = pm.TruncatedNormal('sigma_stop', mu=mu_sigma_stop, sigma=sigma_sigma_stop, lower=1)
        tau_stop = pm.TruncatedNormal('tau_stop', mu=mu_tau_stop, sigma=sigma_tau_stop, lower=1)

        p_tf_probit = pm.TruncatedNormal('p_tf_probit', mu=mu_p_tf, sigma=sigma_p_tf, lower=-6, upper=6)
        p_tf = pm.Deterministic('p_tf', pm.invprobit(p_tf_probit))
    
    # Draw N samples from the prior distribution
    draws = pm.draw([mu_go, sigma_go, tau_go, mu_stop, sigma_stop, tau_stop, p_tf], 
                    draws=N, random_seed=SEED)
    
    # Initialize a DataFrame to store true parameters
    true_parameters = pd.DataFrame(columns=[
        'participant_id', 'mu_go', 'sigma_go', 'tau_go', 
        'mu_stop', 'sigma_stop', 'tau_stop', 'p_tf'])

    # Conditional execution based on input
    all_trials = []
    p_tf = 0.2
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

    # Save simulated data and true parameters
    simulated_data = pd.concat(all_trials, ignore_index=True)
    file_name = f"simulation/data/hierarchical_simulated_data_{args.type}_SSD.csv"
    simulated_data.to_csv(file_name, index=False)
    print(f"Saved simulated data ({args.type} SSD) for {N} participants with {T} trials each.")

    true_parameters_file_name = f"simulation/param/hierarchical_true_parameters_{args.type}_SSD.csv"
    true_parameters.to_csv(true_parameters_file_name, index=False)
    print(f"Saved true parameters ({args.type} SSD) for {N} participants with {T} trials each.")


if __name__ == '__main__':
    main()

# Example usage:
# python simulate_hierarchical_pymc.py --type fixed
# python simulate_hierarchical_pymc.py --type staircase

# whether to store simluated parameters?