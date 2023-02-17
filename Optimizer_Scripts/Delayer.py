# Delayer.py

import numpy as np
import time

class Delayer:
    """Class that performs delayed or undelayed optimization on the given 
    function with the given optimizer. Simplified version of the Delayer class
    with added functionality of DelayType object parsing and generator delays.
    """
    def __init__(self, delay_type, loss_func, optimizer, save_state=False, 
                 save_loss=False, save_grad=False):
        """Initializer for the Delayer class
            
        Parameters: 
            delay_type (DelayType): object containing delay parameters
            loss_func (class object): the loss function to be optimized over, 
                including dimension n, and loss and gradient functions
            optimizer (class object): an initialized class object that is an 
                optimizer with __call__() that updates that state (takes a 
                state and a state derivative w.r.t the loss_function)
            save_state (bool/tuple): dimensions of the state vector to save at
                each iteration. True for all, False for none.
            save_loss (bool): whether to save loss values at each iteration
            save_grad (bool): whether to save grad values at each iteration
        """
        self.delay_type = delay_type
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.save_state = save_state
        self.save_loss = save_loss
        self.save_grad = save_grad
        
        
    def initialize(self, x_init):
        self.optimizer.initialize(x_init)
        self.time_series = np.tile(x_init, (self.delay_type.max_L+1, 1))
        
        if self.save_state is not False:
            self.state_list = [x_init]
        if self.save_loss is True:
            self.loss_list = [self.loss_func.loss(x_init)]
        if self.save_grad is True:
            self.grad_list = [self.loss_func.grad(x_init)]
    
        
    def update(self, i, D):
        """Called by the optimize method and adds the delay and computes the 
        state to do computations on
  
        Parameters:
           i (int): the value of the iteration of compute_time_series
           D (ndarray, (n,)): 1D array of delay lengths for each coordinate
           
        Returns: 
           new_state (ndarray, (n,)): next state after delayed computations
        """
        # Delay the state and save grad
        del_state = np.diag(self.time_series[D])
        x_grad = self.loss_func.grad(del_state)  
        if self.save_grad is True:
            self.grad_list.append(np.linalg.norm(x_grad)) 
        
        # Update!
        new_state = self.optimizer(del_state, x_grad, i)    
        self.add_new_state(new_state)   
        
        return new_state                                       
        
         
    def add_new_state(self, new_state):
        """Add the new state to the list and roll time series forward."""
        if self.save_state is not False:
            self.state_list.append(new_state)
        
        # Roll forward the delay state array
        self.time_series = np.roll(self.time_series, 1, axis=0)
        self.time_series[0] = new_state
        
        
    def optimize(self, x_init, maxiter=2000, tol=1e-5, break_opt=True):
        """Computes the time series using the passed optimizer from __init__, 
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
        
        class Result:
            """A Result object containing optimization data returned by the 
            Delayer class.
            """
            def __init__(self, delayer, converged, runtime):
                """Save requested values"""
                if delayer.save_state is not False:  
                    self.state_vals = np.asarray(delayer.state_list)
                    if delayer.save_state is not True:  # Extract state dimensions
                        self.state_vals = self.state_vals[:,delayer.save_state]
                        
                if delayer.save_loss is True:
                    self.loss_vals = np.asarray(delayer.loss_list)
                    
                if delayer.save_grad is True:
                    self.grad_vals = np.asarray(delayer.grad_list)
                    
                self.converged = converged
                self.runtime = runtime
        
        
        # Initialize
        self.initialize(x_init)
        D_gen = self.delay_type.D_gen(self.loss_func.n)
        new_state = x_init
        conv = False
        
        # Iterate / optimization
        start = time.time()
        for i in range(1, maxiter+1):
            old_state = new_state                   # Keep the old state
            new_state = self.update(i, next(D_gen)) # Update!
            
            if self.save_loss is True:              # Save loss value
                self.loss_list.append(self.loss_func.loss(new_state))
              
            # Stopping condition for convergence
            if conv is False and np.linalg.norm(new_state - old_state) < tol:  
                conv = True
                if break_opt is True:               # Break optimization?
                    break
    
        runtime = time.time() - start
        
        return Result(self, conv, runtime)
            
        
        
    
