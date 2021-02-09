import numpy as np
from hyperopt import hp, tpe, fmin, Trials
from mpi4py import MPI
from Optimizer_Scripts.param_optimizer_helper import use_adam, use_momentum, use_nesterov, use_rast, use_ackley, test_builder
import dill
import copy

def param_optimizer(args):
    delayer = args['delayer']
    use_delays = args['use_delays']
    maxiter = args['maxiter']
    tol = args['tol']
    range_vals = args['range_vals']
    max_evals = args['max_evals']
    space_search = args['space_search']
    symmetric_delays = args['symmetric_delays']
    constant_learning_rate = args['constant_learning_rate']
    early_stopping = args['early_stopping']
    def gen_learning_rates(params):
        learning_rates = np.zeros(maxiter)
        max_alpha = params['max_learning_rate']
        min_alpha = params['min_learning_rate']
        step_size = params['step_size']
        num_changes = maxiter // (2*step_size)
        current_max = max_alpha
        step_int = (current_max - min_alpha) / step_size
        it_num = 0
        while (it_num < maxiter):
            if (it_num == 0):
                learning_rates[it_num] = min_alpha
            else:
                learning_rates[it_num] = learning_rates[it_num-1] + step_int
            it_num += 1
            #check to update parameters for learning rate updates
            if (it_num % step_size == 0):
                if (it_num % (step_size*2) == 0):           
                    curent_max = current_max/2
                    step_int = (current_max - min_alpha) / step_size
                else:
                    step_int = -1*step_int 
        return learning_rates
        
    def test(delayer, x_init):
        delayer.x_init = x_init
        delayer.compute_time_series(use_delays=use_delays, maxiter=maxiter, tol=tol, symmetric_delays=symmetric_delays)  
        if (early_stopping is False):
            if (delayer.conv is True):
                return delayer.final_val
            else:
                return 1000 * delayer.n
        else:
            #calculate the functional value of each state
            values = [delayer.loss_function(delayer.time_series[i,:]) for i in range(len(delayer.time_series))]
            return np.min(values)      
    def objective(params):
        COMM = MPI.COMM_WORLD
        #reset delayer values
        if (constant_learning_rate is True):
            delayer.Optimizer.params['learning_rate'] = params['learning_rate'] * np.ones(maxiter)
        else:
            delayer.Optimizer.params['learning_rate'] = gen_learning_rates(params)
        if (COMM.rank == 0):
            job_vals = np.random.uniform(range_vals[0],range_vals[1],(COMM.size,n))
            job_vals = np.vsplit(job_vals, COMM.size)
        else:
            job_vals = None
        job_vals = COMM.scatter(job_vals, root=0)
        opt_vals = []
        for i in job_vals:
            opt_val = test(delayer=copy.deepcopy(delayer), x_init=i)
            opt_vals.append(opt_val)
        opt_vals = MPI.COMM_WORLD.gather(opt_vals, root = 0)
        if (COMM.rank == 0):
            results = np.asarray([_opt_val for temp in opt_vals for _opt_val in temp])
            final_val = np.mean(results)
        else:
            final_val = None
        final_val = COMM.bcast(final_val)
        return final_val
            
    space_search = space_search
    best = fmin(fn = objective, space=space_search, algo=tpe.suggest, max_evals=max_evals)
    search_options = np.arange(10,2500,10)
    if (constant_learning_rate is False):
        best['step_size'] = search_options[best['step_size']]
    return best, delayer
             
if __name__ == "__main__":
    n = 10000
    max_L = 1
    num_delays = 1000
    use_delays = False
    symmetric_delays = True
    maxiter = 5000
    tol = 1e-5
    optimizer_name = 'Adam'
    loss_name = 'Ackley'
    max_evals=300
    constant_learning_rate = False
    early_stopping = True
    #build the tester
    args = test_builder(n, max_L, num_delays, use_delays, maxiter, optimizer_name, loss_name, tol, max_evals, symmetric_delays, constant_learning_rate, early_stopping)
    #now choose which one to use
    for i in range(10):
        best_params, delayer = param_optimizer(args)
        COMM = MPI.COMM_WORLD
        #save the results
        if (COMM.rank == 0): 
            if (constant_learning_rate is True):
                delayer.Optimizer.learning_rate_bounds = best_params['learning_rate']
            else:
                delayer.Optimizer.learning_rate_bounds = best_params
            print(delayer.Optimizer.learning_rate_bounds)
            print(delayer.Optimizer.name)
            with open('../results/del_{}/lr_{}/test_sym{}_{}_{}_{}_stop_{}_{}.pkl'.format(use_delays, constant_learning_rate, symmetric_delays, optimizer_name, loss_name, delayer.n, early_stopping, i),'wb') as inFile:
                dill.dump(delayer,inFile)      
            del delayer
            
