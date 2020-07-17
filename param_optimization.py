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
    def test(delayer, x_init):
        delayer.x_init = x_init
        delayer.compute_time_series(use_delays=use_delays, maxiter=maxiter, tol=tol, symmetric_delays=symmetric_delays)  
        if (delayer.conv is True):
            return delayer.final_val
        else:
            return 1000 * delayer.n           
    def objective(params):
        COMM = MPI.COMM_WORLD
        #reset delayer values
        delayer.Optimizer.params['learning_rate'] = params['learning_rate']
        if (COMM.rank == 0):
            job_vals = np.random.uniform(range_vals[0],range_vals[1],(2*COMM.size,n))
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
    return best, delayer
             
if __name__ == "__main__":
    n = 100
    max_L = 1
    num_delays = 1000
    use_delays = True
    symmetric_delays = False
    maxiter = 5000
    tol = 1e-5
    optimizer_name = 'Adam'
    loss_name = 'Ackley'
    max_evals=200
    #build the tester
    args = test_builder(n, max_L, num_delays, use_delays, maxiter, optimizer_name, loss_name, tol, max_evals, symmetric_delays)
    #now choose which one to use
    for i in range(10):
        best_params, delayer = param_optimizer(args)
        COMM = MPI.COMM_WORLD
        #save the results
        if (COMM.rank == 0): 
            delayer.Optimizer.params['learning_rate'] = best_params['learning_rate']
            print(delayer.Optimizer.params)
            print(delayer.Optimizer.name)
            with open('../results/{}/test_{}_{}_{}_{}_{}.pkl'.format(use_delays, symmetric_delays, optimizer_name, loss_name, delayer.n, i),'wb') as inFile:
                dill.dump(delayer,inFile)      
            del delayer
            
