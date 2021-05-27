# delay-optimizer
This project deals with adding time delays to an optimizer given some cost function and its gradient.
The different available optimizers are in Optimizer_Scripts/optimizers.py.
The delayer is found in Optimizer_Scripts/Delayer.py
hyperparameter_optimization.py is used for finding optimal hyperparameters for the rastrigin or ackley functions.

## Adding time delays to an optimizer
You can use a generic cost function and optimizer using time delays by passing the function, its gradient, the type of optimizer to delay, and additional hyperparameters to the Delayer class found in Optimizer_Scripts/Delayer.py.
Then call Delayer.compute_time_series() with the relevant hyperparameters.
unit_test.py is outdated.
