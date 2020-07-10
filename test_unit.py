import pytest
import numpy as np
from autograd import grad
from autograd.misc.optimizers import adam
from Optimizer_Scripts.functions import ackley_gen, rastrigin_gen, ackley_deriv_gen, rast_deriv_gen, poly, poly_1, poly_1_grad
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
    n_list = [2,5,10,100,150,200]
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
            if (j == 0):
                x_init = np.zeros(n,dtype=float)
            elif (j == 97):
                x_init = 5.12 * np.ones(n)
                x_init[:len(x_init):2] = -5.12
            elif (j == 98):
                x_init = 5.12 * np.ones(n)
            elif (j == 99):
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
            if (j == 0):
                x_init = 1e-18*np.ones(n,dtype=float)
            elif (j == 97):
                x_init = 32. * np.ones(n)
                x_init[:len(x_init):2] = -32.
            elif (j == 98):
                x_init = 32. * np.ones(n)
            elif (j == 99):
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
            
def test_delay():
    #simple 2-dimensional case with momentum delayed
    
    #declare the optimizer and initialize the Delayer
    series_test = np.array([[16.,8.],[8.,4.],[4.,2.],[2.,-1.],[1.,-1.5]])
    optimizer = Momentum(learning_rate=0.25,gamma=0.5)
    objective = poly
    my_opt = Delayer(n=2, x_init = np.array([16.,8.]), optimizer=optimizer, loss_function=objective, grad=grad(objective), max_L=4, num_delays=50)
    my_opt.compute_time_series(use_delays=True, maxiter=4, random=False, D=np.array([4,3,2,1]))
    assert np.allclose(series_test, my_opt.time_series), "Did not get correct time series on Momentum, wrong state is {}".format(my_opt.time_series)
    assert np.allclose(np.array([1,-1.5]),my_opt.final_state), "Did not get final state on Momentum, wrong state is {}".format(my_opt.final_state)
    
    #simple 2-dimensional case with nesterov momentum delayed
    series_test = np.array([[16.,8.],[8.,4.],[6.,3.],[5.5,0.75],[5.375,0.6875]])
    optimizer = NesterovMomentum(learning_rate=0.25,gamma=0.5)
    objective = poly
    my_opt = Delayer(n=2, x_init = np.array([16.,8.]), optimizer=optimizer, loss_function=objective, grad=grad(objective), max_L=4, num_delays=50)
    my_opt.compute_time_series(use_delays=True, maxiter=4, random=False, D=np.array([4,3,2,1]))
    assert np.allclose(series_test,my_opt.time_series), "Did not get correct time series on NesterovMomentum, wrong state is {}".format(my_opt.time_series)
    assert np.allclose(np.array([5.375,0.6875]),my_opt.final_state), "Did not get final state on NesterovMomentum, wrong state is {}".format(my_opt.final_state)
    
    #same as first test but different delays
    #declare the optimizer and initialize the Delayer
    series_test = np.array([[16.,8.],[8.,4.],[4.,2.],[-2.,-1.],[-3.,-1.5]])
    optimizer = Momentum(learning_rate=0.25,gamma=0.5)
    objective = poly
    my_opt = Delayer(n=2, x_init = np.array([16.,8.]), optimizer=optimizer, loss_function=objective, grad=grad(objective), max_L=3, num_delays=50)
    my_opt.compute_time_series(use_delays=True, maxiter=4, random=False, D=np.array([1,3,2,1]))
    assert np.allclose(series_test, my_opt.time_series), "Did not get correct time series on Momentum 2, wrong state is {}".format(my_opt.time_series)
    assert np.allclose(np.array([-3.,-1.5]),my_opt.final_state), "Did not get final state on Momentum 2, wrong state is {}".format(my_opt.final_state)
    del my_opt
    
    
def test_delay_2():
    #more complicated 2-d case
    
    series_test = np.array([[2.,1.],[-1.,-4.],[-5/2,-13/2],[-7/4,29/4],[7/8,43/8]])
    optimizer = Momentum(learning_rate=0.25,gamma=0.5)
    objective = poly_1
    my_opt = Delayer(n=2, x_init=np.array([2.,1.]), optimizer=optimizer, loss_function=objective, grad=poly_1_grad, max_L=3, num_delays=50)
    my_opt.compute_time_series(use_delays=True, maxiter=4, random=False, D=np.array([1,3,2,1]))
    assert np.allclose(series_test, my_opt.time_series), "Did not get correst time series on Momentum 3, wrong state is {}".format(my_opt.time_series)
    assert np.allclose(np.array([7/8,43/8]), my_opt.final_state)
    del my_opt

#write the final test for checking if the delays are then set to zero
"""
def test_delay_to_zero():
    x_init = np.random.uniform(-5.12,5.12,5) 
    learning_rate = 0.001
    beta_1 = 0.72
    beta_2 = 0.84
    epsilon = 1e-5
    optimizer = Adam(learning_rate=learning_rate, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)
    objective = rastrigin_gen(5)
    gradient = grad(objective)
    my_opt = Delayer(n=5, x_init=x_init, optimizer=optimizer, loss_function=objective, grad=gradient, max_L=1, num_delays=10)
    my_opt.compute_time_series(use_delays=True, maxiter=60, random=True)
    print(len(my_opt.time_series))
    print(my_opt.time_series)
    #get the right state to initialize the starting point of the adam optimizer tester
    x_init_test = my_opt.time_series[10,:]    
    state_test = adam(grad=gradient, x0=x_init_test, num_iters=40, step_size = learning_rate, b1=beta_1, b2 = beta_2, eps = epsilon)
    state_mine = my_opt.final_state
    assert np.allclose(state_test, state_mine), "Fail, my state is {}: and their state is {}".format(state_mine, state_test)
"""    
    
