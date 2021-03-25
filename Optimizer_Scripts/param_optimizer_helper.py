from Optimizer_Scripts.functions import ackley_gen, rastrigin_gen, ackley_deriv_gen, rast_deriv_gen
from Optimizer_Scripts.optimizers import Adam, Momentum, NesterovMomentum
from Optimizer_Scripts.Delayer import Delayer
from Optimizer_Scripts.combustion import get_combustion_model
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
    
def use_rast(n,max_L,num_delays,optimizer, constant_learning_rate,logging):
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
        'max_learning_rate': hp.uniform('max_learning_rate', 1.5, 4.0),
        'min_learning_rate': hp.uniform('min_learning_rate', 0.0, 1.0),
        'step_size': hp.choice('step_size', np.arange(100,2500,100))
        }
    
    delayer = Delayer(n, optimizer, loss_function, deriv_loss, x_init, max_L, num_delays, logging)
    return delayer, space_search, -5.12, 5.12
    
def use_ackley(n,max_L,num_delays,optimizer,constant_learning_rate,logging):
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
        'max_learning_rate': hp.uniform('max_learning_rate', 1.5, 4.0),
        'min_learning_rate': hp.uniform('min_learning_rate', 0.0, 1.0),
        'step_size': hp.choice('step_size', np.arange(100,2500,100))
        }
    delayer = Delayer(n, optimizer, loss_function, deriv_loss, x_init, max_L, num_delays)
    return delayer, space_search, -32., 32.
    
def use_combustion(n,max_L,num_delays,optimizer,constant_learning_rate,vary_percent,logging):
    #get the combustion model values
    x_min, x_init, objective, gradient = get_combustion_model(vary_percent=vary_percent)
    #define the space to search for optimizer hyperparameters
    if (constant_learning_rate is True):
        space_search = {
        'learning_rate': hp.uniform('learning_rate', 0.0, 1.5),
        }
    else:
        space_search = {
        'max_learning_rate': hp.uniform('max_learning_rate', 1.0, 3.0),
        'min_learning_rate': hp.uniform('min_learning_rate', 0.0, 1.0),
        'step_size': hp.choice('step_size', np.arange(10,500,10))
        }
    delayer = Delayer(n, optimizer, objective, gradient, x_init, max_L, num_delays, logging)
    return delayer, space_search, x_min    
    
def test_builder(args):
     #first define the optimizer
     n = args['dim']
     max_L = args['max_L']
     num_delays = args['num_delays']
     constant_learning_rate = args['constant_learning_rate']
     if (args['optimizer_name'] == 'Adam'):
         optimizer = use_adam(epsilon=1e-7, params={'learning_rate': 0.1, 'beta_1': 0.9, 'beta_2': 0.999})
     elif (args['optimizer_name'] == 'Momentum'):
         optimizer = use_momentum(params={'gamma':0.6, 'learning_rate':0.1}, constant_learning_rate=constant_learning_rate)
     else:
         optimizer = use_nesterov(params={'gamma':0.6, 'learning_rate':0.1}) 
     #now define the loss function
     if (args['loss_name'] == 'Rastrigin'):
         args['delayer'], args['space_search'], args['min_val'], args['max_val']=use_rast(n, max_L, num_delays, optimizer, constant_learning_rate, args['logging'])
     elif (args['loss_name'] == 'Ackley'):
         args['delayer'], args['space_search'], args['min_val'], args['max_val']=use_ackley(n, max_L, num_delays, optimizer, constant_learning_rate, args['logging'])
     elif (args['loss_name'] == 'Combustion'):
         args['delayer'], args['space_search'], args['minimizer'] = use_combustion(n, max_L, num_delays, optimizer, constant_learning_rate, args['vary_percent'], args['logging'])
     else:
         raise ValueError("Not a valid objective function use {Rastrigin, Ackley, or Combustion}")
     #now return all the needed parameters as a dictionary
     return args
