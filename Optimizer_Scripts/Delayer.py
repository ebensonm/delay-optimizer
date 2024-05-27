# Delayer.py

import numpy as np
import time

class Delayer:
    """Class that performs delayed or undelayed optimization on the given 
    function with the given optimizer. Simplified version of the Delayer class
    with added functionality of DelayType object parsing and generator delays.
    """
    def __init__(self, delay_type, loss_func, optimizer, save_state=False, 
                 save_loss=False, save_grad=False, full_delay=False):
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
        self.full_delay = full_delay

        if full_delay and self.optimizer.name != "Adam":
            raise NotImplementedError("Only the Adam optimizer is supported for full delays.")

        
    def initialize(self, x_init):
        self.optimizer.initialize(x_init)
        self.time_series = np.tile(x_init, (self.delay_type.max_L+1, 1)).astype(float)
        self.m_series = np.tile(self.optimizer.m_t, (self.delay_type.max_L+1, 1)).astype(float)
        self.v_series = np.tile(self.optimizer.v_t, (self.delay_type.max_L+1, 1)).astype(float)
        
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
        x_grad = self.loss_func.grad(del_state)             # TODO: Gradient saving may be delayed by an index

        if self.full_delay:
            # Delay the optimizer's m_t and v_t
            self.optimizer.m_t = np.diag(self.m_series[D])
            self.optimizer.v_t = np.diag(self.v_series[D])
        
        # Update!
        new_state = self.optimizer(del_state, x_grad, i)    

        # Roll forward the time series and add new state
        self.time_series = np.roll(self.time_series, 1, axis=0)
        self.time_series[0] = new_state  

        if self.full_delay:
            # Roll forward the optimizer's m_t and v_t
            self.m_series = np.roll(self.m_series, 1, axis=0)
            self.m_series[0] = self.optimizer.m_t
            self.v_series = np.roll(self.v_series, 1, axis=0)
            self.v_series[0] = self.optimizer.v_t

        # Log / save values
        self.log(state=new_state, grad=x_grad, loss=self.loss_func.loss(new_state))
        
        return new_state                                       
        

    def log(self, state, grad, loss):
        if self.save_state is not False:
            self.state_list.append(state)
        if self.save_grad is True:
            self.grad_list.append(grad) 
        if self.save_loss is True:
            self.loss_list.append(loss)
    
        
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
                    self.state_vals = np.asarray(delayer.state_list, dtype=float)
                    if delayer.save_state is not True:  # Extract state dimensions
                        self.state_vals = self.state_vals[:,delayer.save_state]
                        
                if delayer.save_loss is True:
                    self.loss_vals = np.asarray(delayer.loss_list, dtype=float)
                    
                if delayer.save_grad is True:
                    self.grad_vals = np.asarray(delayer.grad_list, dtype=float)
                    
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
              
            # Stopping condition for convergence
            if conv is False and np.linalg.norm(new_state - old_state) < tol:  
                conv = True
                if break_opt is True:               # Break optimization?
                    break
    
        runtime = time.time() - start
        
        return Result(self, conv, runtime)
            
        
# # For testing
# if __name__ == "__main__":
#     import sys
#     sys.path.append("/home/yungdankblast/DelayedOptimization/")

#     from Optimizer_Scripts.DelayTypeGenerators import Undelayed, Stochastic
#     from Optimizer_Scripts.LossFunc import LossFunc
#     from Optimizer_Scripts.optimizers import Adam
#     from Optimizer_Scripts.learning_rate_generator import constant

#     delayer = Delayer(
#         delay_type=Stochastic(max_L=1, num_delays=100),  
#         loss_func=LossFunc("Ackley", 2), 
#         optimizer=Adam(params={'beta_1':0.9, 'beta_2':0.999, 'learning_rate':constant(0.01)}),
#         save_state=False, 
#         save_loss=False, 
#         save_grad=False,
#         full_delay=True
#     )
#     delayer.optimize(np.array([1,1]), break_opt=False)
    
