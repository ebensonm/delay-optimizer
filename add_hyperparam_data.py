#add_hyperparam_data.py
"""
Write to a csv file from the results of the hyperparamter optimizations
stored in .json files
"""

import pandas as pd
import os
import json
import glob

csv_name = "hyperparameter_var_LR1.csv"

dfs = []
file_list = glob.glob('*.json')                    # Get all .json files
columns = None
frame = pd.DataFrame()  #initialize the data frame
series_list=[]
for filename in file_list:
    with open(filename) as json_file:
        data = json.load(json_file) #read the json file as dictionary 
    
    # Get the params from the best_params column and join to the data
    series_list.append(pd.Series(data))
    #os.remove(filename)

df = pd.concat(series_list,axis=1)   # Get a dataframe of all .json data
if os.path.exists('Results/{}'.format(csv_name)):  #read current data to append to other data
    current_data = pd.read_csv(csv_name, header=None, index_col=0)
    df = pd.concat([df, current_data],axis=1)
    
df.to_csv('Results/{}'.format(csv_name), index=True, header=False)

