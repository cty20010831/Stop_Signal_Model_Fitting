'''
This python script computes deviance averaged across chains for the given model fitting.
'''

import os
import argparse
import re
import pandas as pd

def main():
    # Get the directory where this script is located at
    dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(description='Compute total deviance for one model fitting')
    parser.add_argument('--data', type=str, choices=['test', 'real'], help="For test data or real data")
    parser.add_argument('--data_analysis_name', type=str, help="Name of the directory storing data analysis (model fitting) results")

    # Parse command-line arguments
    args = parser.parse_args()

    # Locate csv files storing devaince
    data_dir = "real_data" if args.data == "real" else "test_data"
    data_analysis_dir = os.path.join(dir, data_dir, args.data_analysis_name)

    # Compute deviance averaged across chains
    deviance_sum = 0
    count = 0

    for fname in os.listdir(data_analysis_dir):
        if re.compile(r"^deviance[0-9]+\.csv$").match(fname):
            data = pd.read_csv(os.path.join(data_analysis_dir, fname), usecols=[0], header=None)
            deviance_sum += data.iloc[:, 0].sum()

            count += 1

    average_deviance = deviance_sum / count

    # Explain the number of rows in deviance{number}.csv
    print("The number of rows in the deviance csv equals to total number of samples minus number of burn-in samples and then divied by thinning.")

    # Report deviance averaged across chains
    print(f"The deviance averaged across chains for {args.data_analysis_name} is {average_deviance}")

if __name__ == '__main__':
    main()