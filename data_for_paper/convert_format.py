'''
This python script is used to convert the real data for paper into the format 
our pymc hierarchical bayesian model expects. 
'''
import os
import pandas as pd
import numpy as np
import re

def main():
    # Get the directory where this script is located at
    dir = os.path.dirname(os.path.abspath(__file__))

    # Specify the path for the real data
    data_dir = os.path.join(dir, "STOP-IT (4 Block Version)")

    # Initialize an empty list to store all DataFrames
    df_list = []

    for filename in os.listdir(data_dir):
        # Retrieve the full absolute path for each txt file
        file_path = os.path.join(data_dir, filename)
    
        # Initialize an empty list to store the data
        data = []

        # Open the file using 'with open'
        with open(file_path, 'r') as file:
            for line in file:
                # Strip any leading/trailing whitespace and split the line by spaces
                line_data = line.strip().split()
                # Append the parsed line data to the list
                data.append(line_data)

        # Use the first list as the column names and the rest as data
        df = pd.DataFrame(data[1:], columns=data[0])

        # Drop unnecessary columns 
        df = df.drop(columns=["stimulus", "re"])

        # Determine types of trials ("go" or "stop")
        df = df.rename(columns={"signal": "trial_type"})
        df['trial_type'] = df['trial_type'].replace({"1": "stop", "0": "go"})

        # Determine ssd (NaN for go trials)
        df['ssd'] = df['ssd'].replace({"0": np.nan})
        df['ssd'] = df['ssd'].astype("float64")

        # Remove go trials with zero response time 
        df = df.drop(df[(df['trial_type'] == "go") & (df['rt'] == "0")].index)

        # Determine observed response time (NaN for successful inhibition trials)
        df = df.rename(columns={"rt": "observed_rt"})
        df['observed_rt'] = df['observed_rt'].replace({"0": np.nan})
        df['observed_rt'] = df['observed_rt'].astype("float64")

        # Determine final outcomes for go and stop trials
        df.loc[df['trial_type'] == "go", 'outcome'] = "go"
        df.loc[df["observed_rt"].isna(), 'outcome'] = "successful inhibition"
        df["outcome"].fillna("stop-respond", inplace=True)

        # This part can be further modified to see if we are interested in whether responses for go trials are correct
        df = df.drop(columns=['correct', 'respons'])

        # Add a column of particpant id
        participant_id = int(re.findall(r'-(\d+)\.txt$', filename)[0])
        df['participant_id'] = participant_id

        # Reorder column
        df = df.reindex(columns=["block", "trial", 'trial_type', 'ssd', 'observed_rt', 'outcome', 'participant_id'])

         # Append the DataFrame to the list
        df_list.append(df)
    
    # Concatenate all DataFrames into a single large DataFrame
    df_all = pd.concat(df_list, ignore_index=True)

    # Sort participant_id from small to large (also keep the ascending order of block and trial)
    df_all.sort_values(by=['participant_id', 'block', 'trial'], inplace=True)
    df_all = df_all.drop(columns=['block', 'trial'])

    # Use factorize() to create sequential numbering based on unique participant IDs
    df_all['participant_id'] = pd.factorize(df_all['participant_id'])[0]

    # Save the data
    df_all.to_csv(os.path.join(dir, "data.csv"), index=False)

if __name__ == '__main__':
    main()

# Example usage:
# python data_for_paper/convert_format.py