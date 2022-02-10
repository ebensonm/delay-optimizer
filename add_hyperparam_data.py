#add_hyperparam_data.py
"""
Write to a csv file from the results of the hyperparamter optimizations
stored in .json files
"""

import pandas as pd
import os
import glob
import sys


def main(search=""):
    # Get files
    file_list = glob.glob(search + '*.json')             # Get all .json files
    
    if len(file_list) == 0:
        sys.exit("No files found.")

    # Ask for user confirmation
    print("Add and sort the following files to hyperparameter data:\n")
    for filename in file_list:
        print(filename)
    confirmation = input("\nContinue? (Y/[N]): ")
    if confirmation.lower() != 'y':                      # If not, terminate
        sys.exit(1)

    csv_name = "hyperparameter_data.csv"
    dfs = []
    
    for filename in file_list:
        data = pd.read_json(filename, orient='index')  # Read the .json file
        
        # Get the params from the best_params column and join to the data
        params = pd.Series(data.loc["best_params"][0]).to_frame().transpose()
        data = data.drop("best_params").transpose().join(params)
        if data.isna().sum().sum() != 0:
            print(filename)
            print(data)
            sys.exit("Data contains nan values")
        dfs.append(pd.DataFrame(data))
    
    df = pd.concat(dfs, ignore_index=True)   # Get a dataframe of all .json data
    
    # Write hyperparameter data to csv file
    if os.path.exists(csv_name):
        old_df = pd.read_csv(csv_name)
        new_df = old_df.append(df)
        new_df.to_csv(csv_name, index=False)
    else:
        df.to_csv(csv_name, index=False)
    
    return 0
    
    
if __name__ == "__main__":
    # Pass in argument for which files to sort
    num_args = len(sys.argv)
    args = list(sys.argv)
    
    if num_args >= 2:
        main(args[1])
    else:
        main()
