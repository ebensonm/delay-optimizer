from julia.api import Julia
jl = Julia(compiled_modules=False)
import numpy as np
from hyperopt import hp, tpe, fmin, Trials
from mpi4py import MPI
from Optimizer_Scripts.param_optimizer_helper import test_builder
from Optimizer_Scripts.arguments import get_arguments
import copy
from functools import partial
import json
import time

def main(**args):
    optimizer_arguments = test_builder(args)
    global final_sum
    for i in range(args['num_tests']):
        final_sum = list()
        start = time.time()
        best_params, best_loss, arg_min = parameter_optimizer(optimizer_arguments)
        end = time.time()
        #start the parallel hyperoptimization
        COMM = MPI.COMM_WORLD
        #remove the delays
        if (COMM.rank == 0):
            #write the code for creating the argument dictionary
            arg_dict  = args.copy()
            arg_dict['best_params'] = best_params
            arg_dict['test_time'] = np.round(end-start,3)
            arg_dict['best_params']['step_size'] = int(arg_dict['best_params']['step_size'])
            arg_dict['best_params']['best_loss'] = best_loss
            arg_dict['percent_converge'] = final_sum[arg_min]
            arg_dict['Num_Initial_Values'] = COMM.size*arg_dict['num_test_initials']
            if (arg_dict['loss_name']=='Combustion'):
                del arg_dict['minimizer']
            del arg_dict['delayer'], arg_dict['space_search']
            #save the resulting dictionary as a json file
            with open(arg_dict['filename']+'{}.json'.format(i), 'w') as fp:
                json.dump(arg_dict, fp, indent=4)            

def parameter_optimizer(args):
    space_search = args['space_search']
    #initiate the partial function for use in the objective
    objective = partial(initial_points_test,args)
    #remove this if we get memory errors (saves the trials to return best loss value as well
    trials = Trials()
    best = fmin(fn = objective, space=space_search, algo=tpe.suggest, max_evals=args['max_evals'],trials=trials)
    arg_min = np.argmin(trials.losses())
    best_loss = trials.losses()[arg_min]
    #delete to save space
    del trials
    #find the optimal hyperparameters from the space searched
    if (args['loss_name']=='Combustion'):
        search_options = np.arange(10,500,10)
    else:
        search_options = np.arange(100,2500,100)
    #handle the case without a constant learning rate
    if (args['constant_learning_rate'] is False):
        best['step_size'] = search_options[best['step_size']]
    return best, best_loss, arg_min 
    
def get_chunks(lis,num):
    for i in range(0,len(lis),num):
        yield lis[i:i+num]
    
def initial_points_test(args,params):
    COMM = MPI.COMM_WORLD
    delayer = args['delayer']
    #reset the learning values in the optimizer
    if (COMM.rank == 0):
        #get the initial values
        if (args['loss_name']=='Rastrigin' or args['loss_name']=='Ackley'):
            job_vals = np.random.uniform(args['min_val'], args['max_val'], 
                                    (COMM.size*args['num_test_initials'],args['dim']))
            #split so we can run this process in parallel
        elif (args['loss_name']=='Combustion'):
            minimizer = args['minimizer']
            job_vals = 1.0 + np.random.uniform(-args['vary_percent'],args['vary_percent'],
                                         (COMM.size*args['num_test_initials'],args['dim']))
            job_vals = job_vals * minimizer
        job_vals = np.vsplit(job_vals, COMM.size)
    else:
        job_vals = None
    #scatter across available workers
    job_vals = COMM.scatter(job_vals,root=0)
    #initialize list to store the otpimal values
    optimal_values = []
    total_conv = []
    for job in job_vals:
        optimal_value, value = run_test(delayer=copy.deepcopy(delayer),x_init=job,args=args,params=params)
        optimal_values.append(optimal_value)
        total_conv.append(value)
    #now gather the results
    COMM.barrier()
    optimal_values = COMM.gather(optimal_values, root=0)
    total_conv = COMM.gather(total_conv, root=0)
    if (COMM.rank == 0):
        #take the average of the tests and return the value
        final_result  = np.mean(np.asarray([_opt_val for temp in optimal_values for _opt_val in temp]))
        final_sum0 = np.sum(np.asarray([val for temp in total_conv for val in temp]))
        final_sum0 = final_sum0/(COMM.size*args['num_test_initials'])
    else:
        final_result = None
        final_sum0 = None
    final_result = COMM.bcast(final_result)
    final_sum0 = COMM.bcast(final_sum0)
    final_sum.append(final_sum0)
    return final_result*(2-final_sum0)
 
def run_test(delayer,x_init,args,params):
    delayer.Optimizer.params['learning_rate'] = generate_learning_rates(args['constant_learning_rate'], params)
    delayer.x_init = x_init
    delayer.compute_time_series(use_delays=args['use_delays'], maxiter=args['maxiter'],tol=args['tol'],
                                symmetric_delays=args['symmetric_delays'],save_time_series=False)
    #delete the computed time series to save space
    delayer.delete_time_series()
    value = 0
    if (delayer.conv is True):
        value = 1
    #get the final functional value of the loss function
    loss_val = delayer.final_val
    if (loss_val is None):
        loss_val = args['dim']*10000
    #if we are minimizing distance compute the distance
    if (args['hyper_minimize']=='distance'):
       loss_val = np.linalg.norm(args['minimizer']-delayer.final_state) 
    return loss_val, value
        
def const_lr_gen(params):
    GO = True
    learning_rate = params.learning_rate
    while GO is True:
       yield learning_rate

def non_const_lr_gen(params):  
    #dummy variable for running the generator
    GO = True
    #counter for the iterations
    counter = 0
    #variable for how many iterations per triangle side
    lr_num = params['step_size']
    min_lr = params['min_learning_rate']
    max_lr = params['max_learning_rate']
    #compute the step size
    dh = (max_lr - min_lr)/lr_num
    #the learning rate starter
    learning_rate = params['min_learning_rate'] - dh
    counter_counter = 0
    while GO is True:
        counter += 1
        if (counter % lr_num == 0):
            counter_counter += 1
            #shrinking triangle part of the changing learning rate
            if (counter_counter % 2 == 0):
                max_lr -= (max_lr-min_lr)/2
                dh = (max_lr - min_lr)/lr_num
                counter_counter = 0
            else:
                dh = dh*-1
            counter = 0
        learning_rate += dh
        yield learning_rate      
        
def generate_learning_rates(constant_lr, params):
    """ Create the learning rate generator for constant and nonconstant learning rates
    """
    if (constant_lr is True):
        return const_lr_gen(params)
    else:
        return non_const_lr_gen(params)
        
if __name__ == "__main__":
    #pass the command line arguments to the main function
    args = get_arguments()
    main(**vars(args))
        
        
    
