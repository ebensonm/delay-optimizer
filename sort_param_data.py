# sort_param_data.py

import pandas as pd
import numpy as np

# Read the data
data = pd.read_csv("hyperparameter_data.csv")

# Sort the data
data = data.sort_values(by=["loss_name", "constant_learning_rate", "use_delays", "dim", "max_L", "best_loss"], ascending=[True, True, True, True, True, True])

# Convert back to .csv
data.to_csv("hyperparameter_data.csv", index=False)


