import pytest
import numpy as np
from autograd import grad
from autograd.misc.optimizers import adam
from Optimizer_Scripts.functions import ackley_gen, rastrigin_gen, ackley_deriv_gen, rast_deriv_gen
from Optimizer_Scripts.optimizers import Adam, Momentum, NesterovMomentum
from Optimizer_Scripts.Delayer import Delayer

def test_derivatives():
    n_list = [1,2,5,10,100,150,200]
    #rastrigin functions
    objective_list = [rastrigin_gen(n) for n in n_list]
    grad_objective_list = [grad(objective_list[i]) for i in range(len(objective_list))]
    grad_list = [rast_deriv_gen(n) for n in n_list]

    #ackley functions
    objective_list_ack = [ackley_gen(n) for n in n_list]
    grad_objective_list_ack = [grad(objective_list_ack[i]) for i in range(len(objective_list_ack))]
    grad_list_ack = [ackley_deriv_gen(n) for n in n_list]
    
    for i in range(len(n_list)):
        n = n_list[i]
        auto_rast_obj = grad_objective_list[i]
        my_rast_obj = grad_list[i]
        auto_ackley_obj = grad_objective_list_ack[i]
        my_ackley_obj = grad_list_ack[i]
        for j in range(100):
            if (j == 0):
                x_test_rast = np.zeros(n,dtype=float)
                x_test_ackley = 1e-18*np.ones(n,dtype=float)
            elif (j == 97):
                x_test_rast = 5.12 * np.ones(n)
                x_test_rast[:len(x_test_rast):2] = -5.12
                x_test_ackley = 32. * np.ones(n)
                x_test_ackley[:len(x_test_ackley):2] = -32.
            elif (j == 98):
                x_test_rast = 5.12 * np.ones(n)
                x_test_ackley = 32. * np.ones(n)
            elif (j == 99):
                x_test_rast = -5.12 * np.ones(n)
                x_test_ackley = -32. * np.ones(n)
            else:
                x_test_rast = np.random.uniform(-5.12,5.12,n)
                x_test_ackley = np.random.uniform(-32.,32.,n)
                
            #test rastrigin
            auto_rast = auto_rast_obj(x_test_rast)
            my_rast = my_rast_obj(x_test_rast)
            #test ackley
            auto_ackley = auto_ackley_obj(x_test_ackley)
            my_ackley = my_ackley_obj(x_test_ackley)
            #assert statements
            assert np.allclose(auto_rast, my_rast), "Failed for x as {} on rastrigin on dimension {}".format(x_test_rast,n)
            assert np.allclose(auto_ackley, my_ackley), "Failed for x as {} on ackley on dimension {}".format(x_test_ackley,n)

def test_adam_rast():
    #get the objective and gradient functions
    n_list = [1,2,5,10,100,150,200]
    objective_list = [rastrigin_gen(n) for n in n_list]
    grad_objective_list = [grad(objective_list[i]) for i in range(len(objective_list))]
    #set the relavant hyperparameters
    max_iter = 50
    learning_rate = 0.001
    beta_1 = 0.72
    beta_2 = 0.84
    epsilon = 1e-5
    for i in range(len(objective_list)):
        objective = objective_list[i]
        n = n_list[i]
        grad_objective = grad_objective_list[i]
        for j in range(100):
            if (i == 0):
                x_init = np.zeros(n,dtype=float)
            elif (i == 97):
                x_init = 5.12 * np.ones(n)
                x_init[:len(x_init):2] = -5.12
            elif (i == 98):
                x_init = 5.12 * np.ones(n)
            elif (i == 99):
                x_init = -5.12 * np.ones(n)
            else:
                x_init = np.random.uniform(-5.12,5.12,n)
            #use the autograd adam optimizers
            state_test = adam(grad=grad_objective, x0=x_init, num_iters=max_iter, step_size = learning_rate, b1=beta_1, b2 = beta_2, eps = epsilon)
            val_test = objective(state_test)
            #now use my optimizer
            adam_optimizer = Adam(learning_rate=learning_rate, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)
            my_opt = Delayer(n=n, x_init=x_init, optimizer=adam_optimizer, loss_function=objective, grad=grad_objective)  
            my_opt.compute_time_series(use_delays=False,maxiter=max_iter)
            state_mine = my_opt.final_state
            final_val = my_opt.final_val
            assert np.allclose(state_test, state_mine,rtol=1e-4,atol=1e-4), "Adam rastrigin fail on initial value {} with {} dimensions".format(x_init,n)
            assert np.allclose(final_val, val_test,rtol=1e-4,atol=1e-4), "Adam rastrigin fail on final evaluation computation"
            del my_opt
            del state_test
                       
def test_adam_ackley():
    #get the objective and gradient functions
    n_list = [1,2,5,10,100,150,200]
    objective_list = [ackley_gen(n) for n in n_list]
    grad_objective_list = [grad(objective_list[i]) for i in range(len(objective_list))]
    #set the relavant hyperparameters
    learning_rate = 0.001
    max_iter = 50
    beta_1 = 0.72
    beta_2 = 0.84
    epsilon = 1e-5
    for i in range(len(objective_list)):
        objective = objective_list[i]
        n = n_list[i]
        grad_objective = grad_objective_list[i]
        for j in range(100):
            if (i == 0):
                x_init = 1e-18*np.ones(n,dtype=float)
            elif (i == 97):
                x_init = 32. * np.ones(n)
                x_init[:len(x_init):2] = -32.
            elif (i == 98):
                x_init = 32. * np.ones(n)
            elif (i == 99):
                x_init = -32 * np.ones(n)
            else:
                x_init = np.random.uniform(-32.,32.,n)
            #use the autograd adam optimizers
            state_test = adam(grad=grad_objective, x0=x_init, num_iters=max_iter, step_size = learning_rate, b1=beta_1, b2 = beta_2, eps = epsilon)
            val_test = objective(state_test)
            #now use my optimizer
            adam_optimizer = Adam(learning_rate=learning_rate, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)
            my_opt = Delayer(n=n, x_init=x_init, optimizer=adam_optimizer, loss_function=objective, grad=grad_objective)  
            my_opt.compute_time_series(use_delays=False,maxiter=max_iter)
            state_mine = my_opt.final_state
            final_val = my_opt.final_val
            assert np.allclose(state_test, state_mine), "Adam ackley fail on initial value {} with {} dimensions".format(x_init,n)
            assert np.allclose(final_val, val_test), "Adam ackley fail on final evaluation computation"
            del my_opt
            del state_test
            
###########################
#to do - compute a simple delay added system and make sure that its answer correspond to the other answers
#maybe do the comparision test using the other system?          
#def test_delay():

###########################
    
