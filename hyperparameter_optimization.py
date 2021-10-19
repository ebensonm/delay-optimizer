import numpy as np
from hyperopt import hp, tpe, fmin, Trials
from mpi4py import MPI
from Optimizer_Scripts.param_optimizer_helper import test_builder
from Optimizer_Scripts.arguments import get_arguments
import copy
from functools import partial
import json
import time
from Optimizer_Scripts.learning_rate_generator import generate_learning_rates


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
            if arg_dict['constant_learning_rate'] is False:
                arg_dict['best_params']['step_size'] = int(arg_dict['best_params']['step_size'])
            arg_dict['best_params']['best_loss'] = best_loss
            arg_dict['percent_converge'] = final_sum[arg_min]
            arg_dict['num_initial_values'] = COMM.size*arg_dict['num_test_initials']
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
        job_vals = np.random.uniform(args['min_val'], args['max_val'], 
                                    (COMM.size*args['num_test_initials'],args['dim']))
        #split so we can run this process in parallel
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
        
if __name__ == "__main__":
    #pass the command line arguments to the main function
    args = get_arguments()
    main(**vars(args))
        
        
    
