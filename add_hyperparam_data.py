#add_hyperparam_data.py
"""
Write to a csv file from the results of the hyperparamter optimizations
stored in .json files
"""

import pandas as pd
import os
import glob

csv_name = "hyperparameter_data.csv"

dfs = []
file_list = glob.glob('*.json')                    # Get all .json files

for filename in file_list:
    data = pd.read_json(filename, orient='index')  # Read the .json file
    
    # Get the params from the best_params column and join to the data
    params = pd.Series(data.loc["best_params"][0]).to_frame().transpose()
    data = data.drop("best_params").transpose().join(params)
    
    dfs.append(pd.DataFrame(data))

df = pd.concat(dfs, ignore_index=True)   # Get a dataframe of all .json data

# Write hyperparameter data to csv file
if os.path.exists(csv_name):
    df.to_csv(csv_name, index=False, header=False, mode='a')
else:
    df.to_csv(csv_name, index=False)

