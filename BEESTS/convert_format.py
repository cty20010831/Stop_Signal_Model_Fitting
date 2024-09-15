'''
This python script is used to convert the data from `simulation` folder into 
the format BEESTS software expects. 
'''
import os
import pandas as pd
import argparse

def main():
    # Get the directory where this script is located at
    dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(description='Convert data format for BEESTS.')
    parser.add_argument('--type', type=str, choices=['individual', 'hierarchical'], 
                        required=True, help='Level of fitting".')
    
    parser.add_argument('--N', type=int, required=True, 
                        help='Number of participants to (starting from the smallest participant_id, i.e., beginning rows of the dataframe).')
    
    parser.add_argument('--ssd', type=str, choices=['fixed', 'staircase'], required=True, 
                        help='Type of data to generate: "fixed" for fixed SSD or "staircase" for staircase SSD.')

    # Parse command-line arguments
    args = parser.parse_args()

    # Specify the path for the original simulated data
    simulation_dir = os.path.join(os.path.dirname(dir), "simulation")
    simulation_data_path = os.path.join(simulation_dir, "simulated_data", "{}_simulated_data_{}_SSD.csv".format(args.type, args.ssd))
    simulated_data = pd.read_csv(simulation_data_path)

    # Select rows based on specified N
    output_data = simulated_data[simulated_data['participant_id'].isin(range(args.N))]
    
    # Rename column names
    output_data.rename(columns={"trial_type": "ss_presented", 
                                "outcome": "inhibited",
                                "participant_id": "subj_idx",
                                "observed_rt": "rt"}, inplace=True)
    # Drop ssrt column
    output_data.drop(columns="ss_rt", inplace=True)
    
    # Adjust the format (level/group names for categorical variables)
    output_data['ss_presented'] = output_data['ss_presented'].replace({'go': 0, 'stop': 1})
    output_data['inhibited'] = output_data['inhibited'].replace({'go': -999, 
                                                                 "stop-respond": 0,
                                                                 "successful inhibition": 1})
    output_data.fillna(-999, inplace=True)

    # Remove subj_idx for individual level fitting
    if args.type == "individual":
        output_data.drop(columns="subj_idx", inplace=True)
    
    # Reorder column
    output_data = output_data.reindex(columns=['subj_idx', 'ss_presented', 'inhibited', 'ssd', 'rt'])
    print(output_data)

    # Save the data
    os.makedirs(os.path.join(dir, 'data'), exist_ok=True)
    output_data.to_csv(os.path.join(dir, f"data/{args.type}_simulated_data_{args.ssd}_SSD.csv"), index=False)

if __name__ == '__main__':
    main()

# Example usage:
# python BEESTS/convert_format.py --type hierarchical --N 9 --ssd fixed