#!/bin/bash
module load python/3.7

echo "Checking for files..."
if python add_hyperparam_data.py $1; then
    echo "Files added successfully."
    echo "Sorting hyperparameters..."
    python get_best_params.py
    echo "Best hyperparameters written to best_hyperparams.csv."
    echo "Moving .json files to folder..."
    mv $1*.json hyperparam_results/
    echo "Files moved successfully."
else
    echo "Operation cancelled."
fi 

