from hyperopt import hp, tpe, fmin, Trials
import numpy as np
from Optimizer_Scripts.functions import ackley_gen, rastrigin_gen, ackley_deriv_gen, rast_deriv_gen
from Optimizer_Scripts.optimizers import Adam, Momentum, NesterovMomentum
from Optimizer_Scripts.Delayer import Delayer

def adam_optimizer_optimizer(delayer,use_delays=True):
    def objective(params):
        alpha = params['alpha']
        beta_1 = params['beta_1']
        #beta_1 = 0.9
        #beta_2 = 0.999
        beta_2 = params['beta_2']
        epsilon = 1e-7
        delayer.Optimizer.learning_rate = alpha
        delayer.Optimizer.beta_1 = beta_1
        delayer.Optimizer.beta_2 = beta_2
        delayer.Optimizer.epsilon = epsilon
        delayer.compute_time_series(use_delays=use_delays)
        if (delayer.conv is True):
            return delayer.final_val
        else:
            return delayer.n * 1000
    space_search = {
        'alpha': hp.uniform('alpha', 0.0, 10.0),
        'beta_1': hp.uniform('beta_1', 0.3, 1.0),
        'beta_2': hp.uniform('beta_2', 0.3, 1.0)
    }
    best = fmin(fn = objective, space=space_search, algo=tpe.suggest, max_evals=1000)
    return best['alpha'], best['beta_1'], best['beta_2']
    
def momentum_optimizer_optimizer(delayer, use_delays=True):
    def objective(params):
        alpha = params['alpha']
        gamma = params['gamma']
        delayer.Optimizer.learning_rate = alpha
        delayer.Optimizer.gamma = gamma
        delayer.compute_time_series(use_delays=use_delays)
        if (delayer.conv is True):
            return delayer.final_val
        else:
            return delayer.n * 1000
    space_search = {
        'alpha': hp.uniform('alpha', 0.0, 10.0),
        'gamma': hp.uniform('gamma', 0.0, 1.0),
    }
    best = fmin(fn = objective, space=space_search, algo=tpe.suggest, max_evals=1000)
    return best['alpha'], best['gamma']
    
def nesterov_optimizer_optimizer(delayer, use_delays=True):
    def objective(params):
        alpha = params['alpha']
        gamma = param['gamma']
        delayer.Optimizer.learning_rate = alpha
        delayer.Optimizer.gamma = gamma
        delayer.compute_time_series(use_delays=use_delays)
        if (delayer.conv is True):
            return delayer.final_val
        else:
            return delayer.n * 1000
    space_search = {
        'alpha': hp.uniform('alpha', 0.0, 10.0),
        'gamma': hp.uniform('gamma', 0.0, 1.0),
    }
    best = fmin(fn = objective, space=space_search, algo=tpe.suggest, max_evals=1000)
    return best['alpha'], best['gamma']
