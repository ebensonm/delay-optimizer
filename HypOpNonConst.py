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
    param = lrg.get_param_dict(lr_type=args['lr_type'])
    total_loss_function = get_loss_function(args, param)
    num_bay_runs = args['num_runs']
    ranges = dict()
    lr_ranges_min = [args['min_range_0'], args['min_range_1']]
    lr_ranges_max = [args['max_range_0'], args['max_range_1']]
    # add values to dictionary containing range values
    ranges['max_lr'] = lr_ranges_max
    ranges['min_lr'] = lr_ranges_min
    p_ranges = [args['p_1'],args['p_2']]
    ranges['p'] = p_ranges 
    gamma_ranges = [args['gamma_1'],args['gamma_2']]
    ranges['gamma'] = gamma_ranges
    step_ranges = [args['step_1'],args['step_2']]
    ranges['step'] = step_ranges
    bayesian_samples = args['bayesian_samples']
    bay_min = run_hyperparameter_optimization(total_loss_function, num_bay_runs,
                                                        ranges, bayesian_samples)
    #create the json file containing the relevant information
    for it, key in enumerate(param):
        args[key] = bay_min[it]
    args['bay_cost'] = total_loss_function(bay_min)
    with open(args['filename']+'.json', 'w') as fp:
                json.dump(args, fp, indent=4) 
    
    print("File saved at {} in the current directory".format(args['filename']+'.json'))  
     

def get_loss_function(args, param):
    num_initials = args['num_initials']
    n = args['dim']
    cost_function,grad,domain_0,domain_1 = get_cost_function(args['cost_function'], n)
    max_delay = args['max_delay']
    use_delays = args['use_delays']
    num_processes = args['num_processes']
    num_delays = args['num_delays']
    maxiter = args['maxiter']
    tol = args['tol']
    lr_type = args['lr_type']
    
    
    def get_adam_optimizer(params):
        for it, key in enumerate(param):
            param[key] = params[it]
        opt_params = dict()
        opt_params['learning_rate'] = lrg.generate_learning_rates(lr_type, param)
        opt_params['beta_1'] = 0.9
        opt_params['beta_2'] = 0.999
        return Adam(opt_params)
        
        
    def total_loss_function(params):
        pool = Pool()
        params = np.ravel(params)
        errors = pool.map(partial_loss_function, num_processes*[params])
        error = np.mean([errors])
        return error


    def partial_loss_function(params):
        x_inits = np.random.uniform(domain_0, domain_1, (num_initials, n))
        error_value = 0
        for init in x_inits:
            optimizer = get_adam_optimizer(params)
            my_opt = Delayer(n=n, x_init=init, optimizer=optimizer,
                             loss_function=cost_function, grad=grad, 
                             max_L=max_delay, num_delays=num_delays)
            my_opt.compute_time_series(use_delays=use_delays,maxiter=maxiter,tol=tol)
            error_value += my_opt.final_val
        error_value = error_value/num_initials
        return error_value
     
    return total_loss_function 


def bayesian_optimization_gpy(loss_function, ranges, num_runs, num_points):
    #get the starting samples and reshape them to fit algorithm
    samples_list = list()
    bds = list()
    for key in ranges:
        value = ranges[key]
        #get values from dictionary to generate samples
        if (value[0] is not None and value[1] is not None):
            samples = np.linspace(value[0], value[1], num_points)
            samples_list.append(samples)
            bds.append({'name':'{}'.format(key),'type':'continuous','domain':value})
    samples_final = np.transpose(np.asarray(samples_list))
    loss_value_samples = np.asarray([loss_function(loss_par) for loss_par in samples_final]).reshape(-1,1)
    #create the fitting kernel and run the algorithm
    kernel = GPy.kern.Matern52(input_dim=len(bds), variance=1.0, lengthscale=1.0)
    optimizer = BayesianOptimization(f=loss_function, 
                                     domain=bds,
                                     model_type='GP',
                                     kernel=kernel,
                                     X_init=samples_final,
                                     Y_init=loss_value_samples,
                                     acquisition_type='EI',
                                     acquisition_jitter=0.05,
                                     noise_va=0.001**2,
                                     exact_feval=False,
                                     normalize_Y=False,
                                     maximize=False)
    optimizer.run_optimization(max_iter=num_runs)
    optimal_value = optimizer.X[np.argmin(optimizer.Y)]
    #plot the results
    #optimizer.plot_acquisition()
    #optimizer.plot_convergence()
    return optimizer.X[np.argmin(optimizer.Y)]
    

def run_hyperparameter_optimization(loss_function, num_runs, ranges, bayesian_samples):
    #initialize multiprocessing technique
    bay_min = bayesian_optimization_gpy(loss_function, ranges, num_runs, bayesian_samples)
    print("Minimum value on Bayesion search is: {}".format(loss_function(bay_min)))
    print("Optimal learning rate on bayesion search is: {}".format(bay_min[0]))
    #return the minimum value of the two searches
    return bay_min


if __name__ == "__main__":
    #pass the command line arguments to the main function
    args = get_arguments()
    main(**vars(args))
