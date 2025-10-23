'''
This python script is used to store mean (across all chains) estimated parameter values for each subject in a csv file. 
For now, it only deals with hierarchical/group-level fitting (could add functionalities for individual-level fitting in future). 
'''

import os
import re
import pandas as pd
import numpy as np
import argparse
from scipy.stats import norm

def main():
     # Get the directory where this script is located at
    dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(description='Store mean estimated parameter values for each subject in a csv file.')
    parser.add_argument('--data', type=str, choices=['test', 'real'], help="For test data or real data")
    parser.add_argument('--data_file_name', type=str, help="Name of the data file name (used for model fitting)")
    parser.add_argument('--data_analysis_name', type=str, help="Name of the directory storing data analysis (model fitting) results")
    parser.add_argument('--with_trigger_failure', action='store_true', help="Include trigger failure model")

    # Parse command-line arguments
    args = parser.parse_args()

    # Locate csv files storing devaince
    data_dir = "real_data" if args.data == "real" else "test_data"
    data_analysis_dir = os.path.join(dir, data_dir, args.data_analysis_name)

    # Initialize a dictionary to store parameter means for each subject
    parameter_means = {}

    # Define parameters of interest
    params = ['mu_go', 'sigma_go', 'tau_go', 'mu_stop', 'sigma_stop', 'tau_stop']
    if args.with_trigger_failure:
        params.append('p_tf')

    # Determine number of subjects
    n_subjects = pd.read_csv(os.path.join(dir, data_dir, args.data_file_name)).loc[:, 'subj_idx'].nunique()
    print(f"There are in total {n_subjects} in model fitting.")

    # Initialize a dictionary to accumulate parameter values across chains for each subject
    accumulated_params = {f"subject_{subj + 1}": [] for subj in range(n_subjects)}

    # Traverse the current directory to find matching parameter files
    for fname in os.listdir(data_analysis_dir):
        if re.compile(r"^parameters[0-9]+\.csv$").match(fname):
            file_path = os.path.join(data_analysis_dir, fname)

            # Read the parameter values from the CSV file
            data = pd.read_csv(file_path, sep=';', header=0)  
            
            # For each subject, compute the mean of parameters
            for subj in range(1, n_subjects + 1):
                sub_param_name = [f"{i}_subjpt.{subj}" if i == "p_tf" else f"{i}_subj.{subj}" for i in params]
                subject_params = data[sub_param_name]

                # Convert parameters in probit scale (in this case, p_tf) to probabilities
                if 'p_tf' in params:
                    subject_params.loc[:, f'p_tf_subjpt.{subj}'] = norm.cdf(subject_params.loc[:, f'p_tf_subjpt.{subj}'])

                # Rename columns to standard names
                subject_params.columns = params 

                # Store the parameter values for this subject across chains
                accumulated_params[f"subject_{subj}"].append(subject_params)

            print(f"Processed {fname}")

    # Initialize a dictionary to store the mean values across chains    
    parameter_means = {}

    # Compute the mean parameter values for each subject across all chains
    for subj, param_data_list in accumulated_params.items():
        combined_data = pd.concat(param_data_list, axis=0)
        mean_params = combined_data[params].mean().to_dict()  # Use only the defined columns

        # Store the mean parameter values in the dictionary
        parameter_means[subj.split("_")[1]] = mean_params

    # Convert the dictionary to a DataFrame for easy export
    means_df = pd.DataFrame.from_dict(parameter_means, orient='index')

    # Save the means to a CSV file
    means_df.to_csv(os.path.join(data_analysis_dir, "subject_param.csv"), index_label="subj_idx")

    print("Saved mean parameter values for each subject.")

if __name__ == '__main__':
    main()

# Example usage:
# python BEESTS/generate_sub_param.py --data real --data_file_name real_data_5.csv --data_analysis_name real_data_5.csv_250410-215049 --with_trigger_failure True
# python BEESTS/generate_sub_param.py --data real --data_file_name real_data_5.csv --data_analysis_name real_data_5.csv_250616-142647
# python BEESTS/generate_sub_param.py --data test --data_file_name test_staircase_SimSST_n_5.csv --data_analysis_name test_staircase_SimSST_n_5.csv_250616-143235