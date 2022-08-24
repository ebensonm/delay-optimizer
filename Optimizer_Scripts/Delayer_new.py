# DelayedOptimizer.py

import numpy as np
import time

class Delayer:
    """Class that performs delayed or undelayed optimization on the given 
    function with the given optimizer. Simplified version of the Delayer class
    with added functionality of DelayType object parsing and generator delays.
    """
    def __init__(self, n, delay_type, loss, grad, optimizer, save_state=False, 
                 save_loss=False, save_grad=False):
        """The initializer for the Delayer class
            
        Parameters - 
            n (int): the dimension of the state vector
            delay_type (DelayType): object containing delay parameters
            loss (func): the loss function to be optimized on
            grad (func): the gradient of the loss function
            Optimizer (class object): an initialized class object that is an 
                optimizer with __call__() that updates that state (takes a 
                state and a state derivative w.r.t the loss_function)
        """
        self.n = n
        self.delay_type = delay_type
        self.loss = loss
        self.grad = grad
        self.Optimizer = optimizer
        self.save_state = save_state
        self.save_loss = save_loss
        self.save_grad = save_grad
        # Create the lists of values to save
        self.state_list = list()
        self.loss_list = list()
        self.grad_list = list()
        
        
    def initialize(self, x_init):
        self.Optimizer.initialize(x_init)
        self.time_series = np.tile(x_init, (self.delay_type.max_L+1, 1))
        if self.save_state is True:
            self.state_list = [x_init]
        if self.save_loss is True:
            self.loss_list = [self.loss(x_init)]
        if self.save_grad is True:
            self.grad_list = [self.grad(x_init)]
    
        
    def update(self, i, D):
        """Called by the compute_time_series method and adds the delay and 
        computes the state to do computations on
  
        Parameters -
           i (int): the value of the iteration of compute_time_series
           D (ndarray, (n,)): 1D array of delay lengths for each coordinate
        Returns - 
           new_state (ndarray, (n,)): next state after delayed computations
        """
        # Delay the state and save grad
        del_state = np.diag(self.time_series[D])
        x_grad = self.grad(del_state)   
        if self.save_grad is True:
            self.grad_list.append(np.linalg.norm(x_grad)) 
        
        # Update!
        new_state = self.Optimizer(del_state, x_grad, i)    
        self.add_new_state(new_state)    
        
        return new_state                                       
        
         
    def add_new_state(self, new_state):
        """Add the new state to the list and roll time series forward."""
        if self.save_state is True:                 # Save state value
            self.state_list.append(new_state)
        
        # Roll forward the delay state array
        self.time_series = np.roll(self.time_series, 1, axis=0)
        self.time_series[0] = new_state
        
        
    def optimize(self, x_init, maxiter=2000, tol=1e-5, break_opt=True):
        """Computes the time series using the passed Optimizer from __init__, 
        saves convergence and time_series which is an array of the states
           
        Parameters - 
            x_init (ndarray, (n,)): the initial state of the calculation
            maxiter (int): the max number of iterations
            tol (float): the tolerance of convergence
            break_opt (bool): a boolean determining whether to end optimization
                after convergence 
        Returns - 
            (ndarray, (n,)): the final state of the optimization
            (bool): convergence boolean
            (float): the total optimization time in seconds
        """  
        # Initialize
        self.initialize(x_init)
        D_gen = self.delay_type.D_gen(self.n)
        new_state = x_init
        conv = False
        
        # Iterate / optimization
        start = time.time()
        for i in range(1, maxiter+1):
            old_state = new_state                   # Keep the old state
            new_state = self.update(i, next(D_gen)) # Update!
            
            if self.save_loss is True:              # Save loss value
                self.loss_list.append(self.loss(new_state))
              
            # Stopping condition for convergence
            if conv is False and np.linalg.norm(new_state - old_state) < tol:  
                conv = True
                if break_opt is True:               # Break optimization?
                    break
                
        self.conv = conv
        self.iters = i
        
        return new_state, conv, time.time() - start
            
    
    def delete_state_list(self):
        """Clear the state list"""
        self.state_list = list()
        
        
    def delete_loss_list(self):
        """Clear the loss list"""
        self.loss_list = list()
        
        
    def delete_grad_list(self):
        """Clear the gradient list"""
        self.grad_list = list()
        
        
    def reset(self):
        """Resets the optimizer by deleting saved values"""
        self.delete_grad_list()
        self.delete_loss_list()
        self.delete_time_series()    
        
        
    
