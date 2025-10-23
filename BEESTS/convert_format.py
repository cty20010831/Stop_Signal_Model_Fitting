'''
This python script is used to convert the data into the format BEESTS software expects. 
'''
import os
import pandas as pd
import argparse
import sys

def main():
    # Get the directory where this script is located at
    dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(description='Convert data format for BEESTS.')
    parser.add_argument('--data', type=str, choices=['test', 'real'], help="For test data or real data")
    parser.add_argument('--type', type=str, choices=['individual', 'hierarchical'], help='Level of fitting".')
    parser.add_argument('--N', type=int,
                        help='Number of participants to (starting from the smallest participant_id, i.e., beginning rows of the dataframe).')
    parser.add_argument('--ssd', type=str, choices=['fixed', 'staircase'],
                        help='Type of data to generate: "fixed" for fixed SSD or "staircase" for staircase SSD.')

    # Parse command-line arguments
    args = parser.parse_args()

    # Check conditional requirements
    if args.data == 'test':
        if not all([args.type, args.N, args.ssd]):
            print("Error: --type, --N, and --ssd are required when --data is 'real'.")
            parser.print_usage()
            sys.exit(1)

    # Specify the path for the original simulated data
    if args.data == "test":
        data_dir = os.path.join(os.path.dirname(dir), "simulation")
        data_path = os.path.join(data_dir, "simulated_data", "{}_simulated_data_{}_SSD.csv".format(args.type, args.ssd))
    elif args.data == "real":
        data_dir = os.path.join(os.path.dirname(dir), "data_for_paper")
        data_path = os.path.join(data_dir, "data.csv")
    
    data = pd.read_csv(data_path)
        
    # Select rows based on specified N for testing data
    if args.N:
        output_data = data[data['participant_id'].isin(range(args.N))] 
    else:
        output_data = data
    
    # Rename column names
    output_data.rename(columns={"trial_type": "ss_presented", 
                                "outcome": "inhibited",
                                "participant_id": "subj_idx",
                                "observed_rt": "rt"}, inplace=True)
    # Drop ssrt column for testing data
    if args.data == "test": 
        output_data.drop(columns="ss_rt", inplace=True)
    
    # Adjust the format (level/group names for categorical variables)
    output_data.loc[:, 'ss_presented'] = output_data.loc[:, 'ss_presented'].replace({'go': 0, 'stop': 1})
    output_data.loc[:, 'inhibited'] = output_data.loc[:, 'inhibited'].replace({'go': -999, 
                                                                 "stop-respond": 0,
                                                                 "successful inhibition": 1})
    output_data.fillna(-999, inplace=True)

    # Remove subj_idx for individual level fitting
    if args.type == "individual":
        output_data.drop(columns="subj_idx", inplace=True)
    else:
        # For hierarchical level fitting, we need to adjust the subj_idx
        output_data.loc[:, 'subj_idx'] = output_data.loc[:, 'subj_idx'] + 1
    
    # Reorder column
    output_data = output_data.reindex(columns=['subj_idx', 'ss_presented', 'inhibited', 'ssd', 'rt'])

    # Save the data
    if args.data == "test":
        os.makedirs(os.path.join(dir, 'test_data'), exist_ok=True)
        output_data.to_csv(os.path.join(dir, f"test_data/{args.type}_simulated_data_{args.ssd}_SSD.csv"), index=False)
    elif args.data == "real":
        os.makedirs(os.path.join(dir, 'real_data'), exist_ok=True)
        if args.N:
            # Note that it is a small sample of data
            output_data.to_csv(os.path.join(dir, f"real_data/real_data_{args.N}.csv"), index=False)
        else:
            output_data.to_csv(os.path.join(dir, "real_data/real_data.csv"), index=False)

if __name__ == '__main__':
    main()

# Example usage:
# python BEESTS/convert_format.py --data test --type hierarchical --N 9 --ssd fixed
# python BEESTS/convert_format.py --data real --N 3
# python BEESTS/convert_format.py --data real