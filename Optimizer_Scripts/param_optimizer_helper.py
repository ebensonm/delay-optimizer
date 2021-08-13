from Optimizer_Scripts.functions import ackley_gen, rastrigin_gen, ackley_deriv_gen, rast_deriv_gen, rosenbrock_gen, rosen_deriv_gen, zakharov_gen, zakharov_deriv_gen
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
    
def use_rast(n,max_L,num_delays,optimizer,constant_learning_rate,print_log,clip_grad,clip_val):
    logging = False
    if (print_log is True):
        logging = True
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
    
    delayer = Delayer(n, optimizer, loss_function, deriv_loss, x_init, max_L, 
                      num_delays, logging, print_log, clipping=clip_grad, clip_val=clip_val)
    return delayer, space_search, -5.12, 5.12
    
def use_ackley(n,max_L,num_delays,optimizer,constant_learning_rate,print_log):
    logging = False
    if (print_log is True):
        logging = True
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
    delayer = Delayer(n, optimizer, loss_function, deriv_loss, x_init, max_L, num_delays, logging, print_log)
    return delayer, space_search, -32., 32. 
    
def use_rosenbrock(n,max_L,num_delays,optimizer,constant_learning_rate,print_log):
    logging = False
    if (print_log is True):
        logging = True
    np.random.seed(12)
    x_init = np.random.uniform(-10.,10.,n)
    loss_function = rosenbrock_gen(n)
    deriv_loss = rosen_deriv_gen(n)
    if (constant_learning_rate is True):
        space_search = {
        'learning_rate': hp.uniform('learning_rate', 0.0, 2.0),
        }
    else:
        space_search = {
        'max_learning_rate': hp.uniform('max_learning_rate', 1.5, 4.0),
        'min_learning_rate': hp.uniform('min_learning_rate', 0.0, 1.0),
        'step_size': hp.choice('step_size', np.arange(100,2500,100))
        }
    
    delayer = Delayer(n, optimizer, loss_function, deriv_loss, x_init, max_L, num_delays, logging, print_log)
    return delayer, space_search, -10., 10.

def use_zakharov(n,max_L,num_delays,optimizer,constant_learning_rate,print_log):
    logging = False
    if (print_log is True):
        logging = True
    np.random.seed(12)
    x_init = np.random.uniform(-10.,10.,n)
    loss_function = zakharov_gen(n)
    deriv_loss = zakharov_deriv_gen(n)
    if (constant_learning_rate is True):
        space_search = {
        'learning_rate': hp.uniform('learning_rate', 0.0, 2.0),
        }
    else:
        space_search = {
        'max_learning_rate': hp.uniform('max_learning_rate', 1.5, 4.0),
        'min_learning_rate': hp.uniform('min_learning_rate', 0.0, 1.0),
        'step_size': hp.choice('step_size', np.arange(100,2500,100))
        }
    
    delayer = Delayer(n, optimizer, loss_function, deriv_loss, x_init, max_L, num_delays, logging, print_log)
    return delayer, space_search, -10., 10.

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
         args['delayer'], args['space_search'], args['min_val'], args['max_val']=use_rast(n, max_L, num_delays, optimizer, constant_learning_rate, args['print_log'], args['clip_grad'], args['clip_val'])
     elif (args['loss_name'] == 'Ackley'):
         args['delayer'], args['space_search'], args['min_val'], args['max_val']=use_ackley(n, max_L, num_delays, optimizer, constant_learning_rate, args['print_log'])
     elif (args['loss_name'] == 'Rosenbrock'):
         args['delayer'], args['space_search'], args['min_val'], args['max_val']=use_rosenbrock(n, max_L, num_delays, optimizer, constant_learning_rate, args['print_log'])
     elif (args['loss_name'] == 'Zakharov'):
         args['delayer'], args['space_search'], args['min_val'], args['max_val']=use_zakharov(n, max_L, num_delays, optimizer, constant_learning_rate, args['print_log'])
     else:
         raise ValueError("Not a valid objective function use {Rastrigin, Ackley, or Combustion}")
     #now return all the needed parameters as a dictionary
     return args
