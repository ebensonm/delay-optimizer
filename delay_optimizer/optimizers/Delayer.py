import numpy as np
import time

class Delayer:
    """Class that performs delayed optimization on the given objective function 
    with the given optimizer. The sequence of delays to use is determined by the 
    DelayType object.
    """
    def __init__(self, objective, optimizer, delay_type, save_state=False, 
                 save_loss=False, save_grad=False):
        """Initializer for the Delayer class
            
        Parameters:
            objective (ObjectiveFunction): the loss function to be optimized over, 
                including dimension n, loss function, and gradient function
            optimizer (Optimizer): an initialized class object to perform the
                optimization. Must have a step() method that updates the state
                given state and gradient values.
            delay_type (DelayType): object containing delay parameters
            save_state (bool/tuple): dimensions of the state vector to save at
                each iteration. True to save all dimensions, False for none.
            save_loss (bool): whether to save loss values at each iteration
            save_grad (bool): whether to save grad values at each iteration
        """
        self.objective = objective
        self.optimizer = optimizer
        self.delay_type = delay_type
        self.save_state = save_state
        self.save_loss = save_loss
        self.save_grad = save_grad
        
    def initialize(self, x_init):
        self.optimizer.initialize(x_init)
        self.time_series = np.tile(x_init, (self.delay_type.max_L+1, 1))
        self.delay_generator = self.delay_type.D_gen(self.objective.n)
        
        if self.save_state is not False:
            self.state_list = [x_init]
        if self.save_loss is True:
            self.loss_list = [self.objective.loss(x_init)]
        if self.save_grad is True:
            self.grad_list = [self.objective.grad(x_init)]
    
    def step(self):
        """Computes the delayed state and gradient values and takes a step
        with the optimizer.
        """
        # Compute the delayed state and gradient
        D = next(self.delay_generator)              # Get the delay
        del_state = np.diag(self.time_series[D])
        del_grad = self.objective.grad(del_state)
        
        # Update!
        new_state = self.optimizer.step(del_state, del_grad)    

        # Roll forward the time series and add new state
        self.time_series = np.roll(self.time_series, 1, axis=0)
        self.time_series[0] = new_state  

        # Log / save values
        # TODO: Saved gradients are the delayed gradient values, may want to recompute gradient or stop saving gradients
        #       Furthermore, the entire gradient is saved, which is usually not done even with state values, so we may
        #       want to compute and save the gradient norm if we want to save gradients still.
        self.log(state=new_state, grad=del_grad, loss=self.objective.loss(new_state))
        
        return new_state
    
    def log(self, state, grad, loss):
        if self.save_state is not False:
            self.state_list.append(state)
        if self.save_grad is True:
            self.grad_list.append(grad) 
        if self.save_loss is True:
            self.loss_list.append(loss)
        
    def optimize(self, x_init, maxiter=5000):
        """Computes the time series using the passed optimizer from __init__, 
        saves convergence and time_series which is an array of the states
           
        Parameters:
            x_init (ndarray): the initial state of the calculation
            maxiter (int): the max number of iterations
        Returns:
            (Result): an object containing the optimization data
        """  
        class Result:
            """An object containing optimization data to be accessed externally"""
            def __init__(self, delayer, runtime):
                """Save requested values"""
                if delayer.save_state is not False:  
                    self.state_vals = np.asarray(delayer.state_list)
                    if delayer.save_state is not True:  # Extract state dimensions
                        self.state_vals = self.state_vals[:,delayer.save_state]
                if delayer.save_loss is True:
                    self.loss_vals = np.asarray(delayer.loss_list)
                if delayer.save_grad is True:
                    self.grad_vals = np.asarray(delayer.grad_list)
                self.runtime = runtime
        
        # Initialize
        self.initialize(x_init) 
        
        # Iterate / optimization
        start = time.time()
        for i in range(maxiter):
            self.step() # The time series automatically updates state values for us
        runtime = time.time() - start
        
        return Result(self, runtime)
            
        
        
    
