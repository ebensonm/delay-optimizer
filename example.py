from Optimizer_Scripts import functions, optimizers
from Optimizer_Scripts.Delayer import Delayer
from Optimizer_Scripts.learning_rate_generator import generate_learning_rates
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi']= 800


if __name__ == "__main__":
    #define the constants
    n = 2
    max_learning_rate = 1.5
    min_learning_rate = 0.1
    step_size = 500
    epsilon = 1e-7
    beta_1 = 0.99
    beta_2 = 0.999
    L=1
    num_delays=1000
    params = dict()
    params['beta_1'] = beta_1
    params['beta_2'] = beta_2
    x_init = np.random.uniform(-32., 32., size=n)
    
    #get the cost and the gradient
    cost = functions.ackley_gen(n)
    cost_gradient = functions.ackley_deriv_gen(n)
    #now generate the learning rate and get the optimizer
    lr_params = dict()
    lr_params['step_size'] = step_size
    lr_params['max_learning_rate'] = max_learning_rate
    lr_params['min_learning_rate'] = min_learning_rate
    params['learning_rate'] = generate_learning_rates(constant_lr = False, params = lr_params)
    optimizer = optimizers.Adam(params=params, epsilon=epsilon)
    #use the Delayer class to add time delays to the optimizer
    delayed_optimizer = Delayer(n, optimizer, cost, cost_gradient, max_L=L, num_delays=num_delays,
                                compute_loss=True, print_log=True, save_grad=True)
    delayed_optimizer.x_init = x_init
    #run the optimizer
    delayed_optimizer.compute_time_series(use_delays=True, save_time_series=True)
    
    #TODO - plot the results of the optimization
    
    time = np.arange(len(delayed_optimizer.grad_list))
    plt.plot(time, delayed_optimizer.grad_list, lw=.5)
    plt.xlabel("Time")
    plt.ylabel("Gradient")
    plt.title("Gradient of the Ackley Function over time")
    plt.show()
    
    plt.plot(time, delayed_optimizer.loss_list, lw=.5)
    plt.xlabel("Time")
    plt.ylabel("Loss Function")
    plt.title("Value of the Ackley Function over time")
    plt.show()
    
    x = delayed_optimizer.time_series[:,0]
    y = delayed_optimizer.time_series[:,1]
    plt.plot(time, x[1:], lw=.5, label='x')
    plt.plot(time, y[1:], lw=.5, label='y')
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Values x and y of the optimization over time")
    plt.show()
    
    plt.plot(x, y, zorder=0)
    plt.scatter(x[0], y[0], label="Initial point", c='forestgreen', zorder=1)
    plt.scatter(x[-1], y[-1], label="Optimizer", c='C1', zorder=1)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Graph of y over x for the optimization")
    plt.show()
