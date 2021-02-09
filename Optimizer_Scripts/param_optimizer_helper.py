from Optimizer_Scripts.functions import ackley_gen, rastrigin_gen, ackley_deriv_gen, rast_deriv_gen
from Optimizer_Scripts.optimizers import Adam, Momentum, NesterovMomentum
from Optimizer_Scripts.Delayer import Delayer
import numpy as np
from hyperopt import hp, tpe, fmin, Trials

def use_adam(params, epsilon=1e-7):
    optimizer = Adam(params, epsilon=epsilon)
    return optimizer
 
def use_momentum(params):
    optimizer = Momentum(params)
    return optimizer
    
def use_nesterov(params):    
    optimizer = NesterovMomentum(params)
    return optimizer
    
def use_rast(n,max_L,num_delays,optimizer, constant_learning_rate):
    np.random.seed(12)
    x_init = np.random.uniform(-5.12,5.12,n)
    loss_function = rastrigin_gen(n)
    deriv_loss = rast_deriv_gen(n)
    if (constant_learning_rate is True):
        space_search = {
        'learning_rate': hp.uniform('learning_rate', 0.0, 1.5),
        }
    else:
        space_search = {
        'max_learning_rate': hp.uniform('max_learning_rate', 1.0, 5.0),
        'min_learning_rate': hp.uniform('min_learning_rate', 0.1, 1.0),
        'step_size': hp.choice('step_size', np.arange(10,2500,10))
        }
    
    delayer = Delayer(n, optimizer, loss_function, deriv_loss, x_init, max_L, num_delays)
    return delayer, space_search, [-5.12,5.12]
    
def use_ackley(n,max_L,num_delays,optimizer,constant_learning_rate):
    np.random.seed(12)            
    x_init = np.random.uniform(-32.,32.,n)
    loss_function = ackley_gen(n)
    deriv_loss = ackley_deriv_gen(n)
    if (constant_learning_rate is True):
        space_search = {
        'learning_rate': hp.uniform('learning_rate', 0.0, 1.5),
        }
    else:
        space_search = {
        'max_learning_rate': hp.uniform('max_learning_rate', 1.0, 5.0),
        'min_learning_rate': hp.uniform('min_learning_rate', 0.1, 1.0),
        'step_size': hp.choice('step_size', np.arange(10,2500,10))
        }
    delayer = Delayer(n, optimizer, loss_function, deriv_loss, x_init, max_L, num_delays)
    return delayer, space_search, [-32.,32.]
    
def test_builder(n, max_L, num_delays, use_delays, maxiter, optimizer_name, loss_name, tol, max_evals, symmetric_delays, constant_learning_rate, early_stopping):
     #first define the optimizer
     if (optimizer_name == 'Adam'):
         optimizer = use_adam(epsilon=1e-7, params={'learning_rate': 0.1*np.ones(maxiter), 'beta_1': 0.9, 'beta_2': 0.999})
     elif (optimizer_name == 'Momentum'):
         optimizer = use_momentum(params={'gamma':0.6, 'learning_rate':0.1*np.ones(maxiter)}, constant_learning_rate=constant_learning_rate)
     else:
         optimizer = use_nesterov(params={'gamma':0.6, 'learning_rate':0.1*np.ones(maxiter)}) 
     #now define the loss function
     if (loss_name == 'Rastrigin'):
         delayer, space_search, range_vals=use_rast(n, max_L, num_delays, optimizer, constant_learning_rate)
     elif (loss_name == 'Ackley'):
         delayer, space_search, range_vals=use_ackley(n, max_L, num_delays, optimizer, constant_learning_rate)
     #now return all parameters in currect order
     return {'delayer':delayer, 'space_search':space_search, 'use_delays':use_delays, 'maxiter':maxiter, 'tol':tol, 'range_vals':range_vals, 'max_evals':max_evals, 'symmetric_delays':symmetric_delays, 'constant_learning_rate': constant_learning_rate, 'early_stopping':early_stopping}
