import numpy as np
import time

class AdaptiveDelayer:

    def __init__(self, n, optimizer, loss_function, adaptive_function, grad, x_init, max_L=2):
        """The initializer for the Delayer class
            
           Parameters - 
               n (int) - the dimension of the state vector
               Optimizer (class object) - an initialized class object that is an optimizer with
               __call__() that updates that state (takes a state and a state derivative w.r.t the loss_function)s
               loss_function (python method) - the loss function to be optimized on
               grad (python method) - the gradient of the loss function
               x_init (ndarray, (n,)) - the initial state of the calculation
               max_L (int) - the maximum delay of the system
               num_delays (int) - the number of delays to compute with the system
        """
        self.n = n
        self.loss_function = loss_function
        self.adaptive_function = adaptive_function
        self.grad = grad
        self.Optimizer = optimizer
        self.x_init = x_init
        self.max_L = max_L
        self.list_n = np.tile(np.arange(0,self.n,1,dtype = int),self.n).flatten()
        self.time_series = list()
        
    def delete_time_series(self):
        """deletes the calculated time series of the compute_time_series method
        """
        self.time_series = list()
        
    def add_copies(self):
        """adds copies to the time series of the initial value to be used for getting delays at the beginning
        """
        self.time_series[:self.max_L+1,:] = self.x_init
    
    def use_delay(self, iter_val, function=True, symmetric_delays=False, D=None):
        if (symmetric_delays is False):
            if (function is True):
                time_del = self.adaptive_function(self.grad, self.time_series[iter_val], self.max_L)
                D = iter_val - np.random.randint(0,time_del+1,self.n**2)  #get list of random delays
            else:
                D = iter_val - D  
            x_state = self.time_series[D, self.list_n].reshape(self.n, self.n)      #use indexing to delay
            x_grad = np.diag(self.grad(x_state - self.Optimizer.grad_helper)) #get the gradient of the delays
            x_state = np.diag(x_state)                                       #get the state to update from     
        else:
            if (function is True):
                time_del = self.adaptive_function(self.grad, self.time_series[iter_val], self.max_L)
                D = iter_val - np.random.randint(0, time_del+1,self.n)
            else:
                D = iter_val - D
            x_state = self.time_series[D, self.list_n[:self.n]]               #use indexing to delay
            x_grad = self.grad(x_state - self.Optimizer.grad_helper)       #get the gradient of the delays
                                       #get the state to update from     
            
        self.x_state = x_state
        self.x_grad = x_grad
        x_state_new = self.Optimizer(x_state, x_grad, iter_val-self.max_L+1) #update!   
        
        return x_state_new
    
        
    def compute_time_series(self, tol=1e-10, maxiter=5000, use_delays=False, random=True, symmetric_delays=False, D=None):
        """computes the time series using the passed Optimizer from __init__, saves convergence
           and time_seris which is an array of the states
           
           Parameters - 
               tol (float) - the tolerance of convergence before ending the optimization
               maxiter (int) - the max number of iterations before determining it did not converge
               use_delays (bool) - whether or not to call the use_delay method which adds delays to
               the state vector
        """
        conv_bool = False                                       #initialize the time series
        self.time_series = np.zeros((maxiter+self.max_L+1,self.n)) #initialize the convergence boolean
        self.num_max_delay = self.max_L                         #initialize number for max delay of iteration
        if (self.Optimizer.initialized is False):
            self.Optimizer.initialize(self.x_init)
        self.add_copies()                 #add copies to the time series for the delay
        for i in range(maxiter):          #start optimizer iterations         
            if (use_delays is True):                                 #computation with delays
                x_state_new = self.use_delay(iter_val = i+self.max_L, D=D, symmetric_delays=symmetric_delays)  #use_delay to get state
            else:                                               #computation without delays
                x_grad = self.grad(self.time_series[i+self.max_L])
                x_state_new = self.Optimizer(self.time_series[i+self.max_L], x_grad, i+1)  
            self.time_series[i+1+self.max_L,:] = x_state_new            #add the updated values to the time series
            if (np.linalg.norm(self.time_series[i+1+self.max_L,:] - self.time_series[i+self.max_L,:]) < tol):
                conv_bool = True
                break
                
        self.Optimizer.initialized = False                    #reset the input optimizer
        self.time_series = self.time_series[self.max_L:i+2+self.max_L,:] #remove copies and end zeros
        self.final_state = x_state_new                        #save the final state
        self.final_val = self.loss_function(self.final_state) #save the final loss value
        self.conv = conv_bool                                 #save convergence boolean
