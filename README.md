# delay-optimizer
This project deals with adding time delays to an optimizer given some cost function and its gradient.
With much experimentation and analysis we have discovered interesting results in many different forms of cost functions.
This method, with some editing can be applied to Machine Learning and Deep Learning Models.
The different available optimizers are in Optimizer_Scripts/optimizers.py.
The delayer is found in Optimizer_Scripts/Delayer.py
hyperparameter_optimization.py is used for finding optimal hyperparameters for the Rastrigin or Ackley functions.

## Adding time delays to an optimizer
You can use a generic cost function and optimizer using time delays by passing the function, its gradient, the type of optimizer to delay, and additional hyperparameters to the Delayer class found in Optimizer_Scripts/Delayer.py.
Then call Delayer.compute_time_series() with the relevant hyperparameters.
Understanding how time delays affect and sometimes improve the optimization process with the Adam Optimizer has already allowed us to improve on optimization test functions found on wikipedia.com.
unit_test.py is outdated.
