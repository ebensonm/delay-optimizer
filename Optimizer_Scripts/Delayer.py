import numpy as np
class Delayer:

    def __init__(self, n, optimizer, loss_function, grad, x_init, max_L, num_delays):
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
        self.grad = grad
        self.Optimizer = optimizer
        self.x_init = x_init
        self.max_L = max_L
        self.list_n = np.tile(np.arange(0,self.n,1,dtype = int),self.n).flatten()
        self.num_delays = num_delays
        
    def delete_time_series(self):
        """deletes the calculated time series of the compute_time_series method
        """
        del self.time_series
        
    def use_delay(self,iter_val):
        """Called up by the compute_time_series method and adds the delay and computes the state/states to
           do computations on
           
           Parameters - 
               iter_val (int) - the value of the iteration of compute_time_series
           returns - 
               x_state_new (ndarray, (n,)) - the next state after delayed computations
        """
        if (iter_val < self.num_delays):                                 #check if we are still adding delays
            if (iter_val % (self.num_delays // self.max_L) == 0):        #shrink delays every interval
                self.num_max_delay -= 1
            D = iter_val - 1 - np.random.randint(0,self.num_max_delay+1,self.n**2)  #get list of random delays
            x_state = self.time_series[D, self.list_n].reshape(self.n, self.n)      #use indexing to delay
            x_grad = np.diag(self.grad(x_state - self.Optimizer.grad_helper))       #get the gradient of the delays
            x_state = np.diag(x_state)                                              #get the state to update from
            x_state_new = self.Optimizer(x_state, x_grad, iter_val)                 #update!     
        else:       
            x_grad = self.grad(self.time_series[iter_val-1] - self.Optimizer.grad_helper)        
            x_state_new = self.Optimizer(self.time_series[iter_val-1], x_grad, iter_val)  
                               
        return x_state_new                                       #return the new state
      
    def compute_time_series(self, tol=1e-10, maxiter=5000, use_delays=False):
        """computes the time series using the passed Optimizer from __init__, saves convergence
           and time_seris which is an array of the states
           
           Parameters - 
               tol (float) - the tolerance of convergence before ending the optimization
               maxiter (int) - the max number of iterations before determining it did not converge
               use_delays (bool) - whether or not to call the use_delay method which adds delays to
               the state vector
        """
        conv_bool = False                                       #initialize the time series
        self.time_series = np.zeros((maxiter,self.n))           #initialize the convergence boolean
        self.num_max_delay = self.max_L                         #initialize number for max delay of iteration
        for i in range(self.max_L+1):                           #add copies to the time series for delays
            self.time_series[i,:] = self.x_init
            iter_start = i + 1    
        for i in range(iter_start,maxiter):                     #start optimizer iterations
            if (use_delays is True):                            #computation with delays
                x_state_new = self.use_delay(iter_val = i)      #use_delay is used and returns the final state
            else:                                               #computation without delays
                x_grad = self.grad(self.time_series[i-1])
                x_state_new = self.Optimizer(self.time_series[i-1], x_grad, i)
                
            self.time_series[i,:] = x_state_new                #add the updated values to the time series
            if (np.linalg.norm(self.time_series[i,:] - self.time_series[i-1,:]) < tol): #check convergences
                conv_bool = True
                break
        self.Optimizer.initialized = False                     #reset the input optimizer
        self.time_series = self.time_series[self.max_L:i+1,:] #remove copies and end zeros from time_series
        self.final_state = x_state_new                        #save the final state
        self.final_val = self.loss_function(self.final_state) #save the final loss value
        self.conv = conv_bool                                 #save convergence boolean
