import numpy as np
import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization
from scipy.stats import norm
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from ray.util.multiprocessing.pool import Pool
from Optimizer_Scripts.optimizers import Adam, Momentum, NesterovMomentum
from Optimizer_Scripts.Delayer import Delayer
from hyperopt_parameters import get_cost_function
from hyperopt_parameters import get_arguments
import json
import Optimizer_Scripts.learning_rate_generator as lrg


def main(**args):
    #get the arguments and run the hyperparameter optimization
    total_loss_function = get_loss_function(args)
    num_bay_runs = args['num_runs']
    ranges = [args['ranges_0'],args['ranges_1']]
    bayesian_samples = args['bayesian_samples']
    grid_samples = args['grid_samples']
    grid_min, bay_min = run_hyperparameter_optimization(total_loss_function, num_bay_runs,
                                                        ranges, bayesian_samples,
                                                        grid_samples)
    #create the json file containing the relevant information
    args['grid_min'] = grid_min
    args['grid_cost'] = total_loss_function(grid_min)
    args['bay_min'] = bay_min
    args['bay_cost'] = total_loss_function(bay_min)
    with open(args['filename']+'.json', 'w') as fp:
                json.dump(args, fp, indent=4) 
    
    print("File saved at {} in the current directory".format(args['filename']+'.json'))  
     

def get_loss_function(args):
    num_initials = args['num_initials']
    n = args['dim']
    cost_function,grad,domain_0,domain_1 = get_cost_function(args['cost_function'], n)
    max_delay = args['max_delay']
    use_delays = args['use_delays']
    num_processes = args['num_processes']
    num_delays = args['num_delays']
    maxiter = args['maxiter']
    tol = args['tol']
    
    
    def get_adam_optimizer(learning_rate):
        param = dict()
        param['learning_rate'] = learning_rate
        opt_params = dict()
        opt_params['learning_rate'] = lrg.const_lr_gen(param)
        opt_params['beta_1'] = 0.9
        opt_params['beta_2'] = 0.999
        return Adam(opt_params)
        
        
    def total_loss_function(learning_rate):
        pool = Pool()
        errors = pool.map(partial_loss_function, num_processes*[learning_rate])
        error = np.mean([errors])
        return error


    def partial_loss_function(learning_rate):
        x_inits = np.random.uniform(domain_0, domain_1, (num_initials, n))
        error_value = 0
        optimizer = get_adam_optimizer(learning_rate)
        for init in x_inits:
            my_opt = Delayer(n=n, x_init=init, optimizer=optimizer,
                             loss_function=cost_function, grad=grad, 
                             max_L=max_delay, num_delays=10000)
            my_opt.compute_time_series(use_delays=use_delays,maxiter=maxiter,tol=tol)
            error_value += my_opt.final_val
        error_value = error_value/num_initials
        return error_value
     
    return total_loss_function 


def grid_search(loss_function, ranges, num_points,tol=1e-3):
    #do a grid search to try to narrow down our hyperparameter optimization search area
    grid, retstep = np.linspace(ranges[0]+tol, ranges[1], num_points, retstep=True)
    loss_values = [loss_function(lr) for lr in grid]
    #return the smallest value
    index = np.argmin(loss_values)
    return grid[index], retstep


def bayesian_optimization_gpy(loss_function, ranges, num_runs, num_points):
    #get the starting samples and reshape them to fit algorithm
    learning_rate_samples = np.linspace(ranges[0],ranges[1],num_points)
    loss_value_samples = np.asarray([loss_function(lr) for lr in learning_rate_samples]).reshape(-1,1)
    learning_rate_samples = learning_rate_samples.reshape(-1,1)
    #create the fitting kernel and run the algorithm
    kernel = GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=1.0)
    bds = [{'name': 'X', 'type': 'continuous', 'domain': ranges}]
    optimizer = BayesianOptimization(f=loss_function, 
                                     domain=bds,
                                     model_type='GP',
                                     kernel=kernel,
                                     X_init=learning_rate_samples,
                                     Y_init=loss_value_samples,
                                     acquisition_type='EI',
                                     acquisition_jitter=0.05,
                                     noise_va=0.01**2,
                                     exact_feval=False,
                                     normalize_Y=False,
                                     maximize=False)
    optimizer.run_optimization(max_iter=num_runs)
    optimal_value = optimizer.X[np.argmin(optimizer.Y)]
    #plot the results
    #optimizer.plot_acquisition()
    #optimizer.plot_convergence()
    return optimizer.X[np.argmin(optimizer.Y)]
    

def run_hyperparameter_optimization(loss_function, num_runs, ranges, bayesian_samples, grid_samples):
    #initialize multiprocessing technique
    grid_min, diff = grid_search(loss_function, ranges, grid_samples) #run the grid search
    print("Minimum value on grid search is: {}".format(loss_function(grid_min)))
    print("Optimal learning rate on grid search is: {}".format(grid_min))
    #get new set of points and refine our search with Bayesian Hyperparameter optimization
    bay_ranges = [np.max([0, grid_min-diff]), grid_min+diff]
    bay_min = bayesian_optimization_gpy(loss_function, bay_ranges, num_runs, bayesian_samples)
    print("Minimum value on Bayesion search is: {}".format(loss_function(bay_min[0])))
    print("Optimal learning rate on bayesion search is: {}".format(bay_min[0]))
    #return the minimum value of the two searches
    return grid_min, bay_min[0]


if __name__ == "__main__":
    #pass the command line arguments to the main function
    args = get_arguments()
    main(**vars(args))


