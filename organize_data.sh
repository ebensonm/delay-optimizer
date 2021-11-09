#!/bin/bash

python add_hyperparam_data.py
python get_best_params.py
mv *.json hyperparam_results

