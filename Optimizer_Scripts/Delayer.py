import numpy as np
import time

class Delayer:

    def __init__(self, n, optimizer, loss_function, grad, x_init, max_L=2, num_delays=None, logging=False):
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
        self.time_series = list()
        self.logging = logging
        if (logging is True):
            self.loss_list = list()
        
    def delete_time_series(self):
        """deletes the calculated time series of the compute_time_series method
        """
        self.time_series = list()
        
    def add_copies(self):
        """adds copies to the time series of the initial value to be used for getting delays at the beginning
        """
        self.time_series[:self.max_L+1,:] = self.x_init
        
    def use_delay(self, index_val, iter_val, random=True, symmetric_delays=False, D=None, shrink=False):
        """Called up by the compute_time_series method and adds the delay and computes the state/states to
           do computations on
  
           Parameters - 
               iter_val (int) - the value of the iteration of compute_time_series
           returns - 
               x_state_new (ndarray, (n,)) - the next state after delayed computations
        """
        if (iter_val < self.num_delays):                                 #check if we are still adding delays
            if (iter_val % (self.num_delays // self.max_L) == 0 and shrink==True):    #shrink delays every interval
                self.num_max_delay -= 1
            if (symmetric_delays is False):
                if (random is True):
                    D = index_val - np.random.randint(0,self.num_max_delay+1,self.n**2)  #get list of random delays
                else:
                    D = index_val - D  
                x_state = self.time_series[D, self.list_n].reshape(self.n, self.n)      #use indexing to delay
                x_grad = np.diag(self.grad(x_state - self.Optimizer.grad_helper)) #get the gradient of the delays
                x_state = np.diag(x_state)                                       #get the state to update from     
            else:
                if (random is True):
                    D = index_val - np.random.randint(0, self.num_max_delay+1,self.n)
                else:
                    D = index_val - D
                x_state = self.time_series[D, self.list_n[:self.n]]               #use indexing to delay
                value = x_state - self.Optimizer.grad_helper
                x_grad = self.grad(value)       #get the gradient of the delays
            #handle the exception case in the combustion problem
            if (x_grad is None):
                return None
            x_state_new = self.Optimizer(x_state, x_grad, iter_val)                 #update!   
        else:
            value = self.time_series[index_val] - self.Optimizer.grad_helper
            x_grad = self.grad(self.time_series[index_val] - self.Optimizer.grad_helper)  
            #handle the exception case in the combustion problem
            if (x_grad is None):
                return None      
            x_state_new = self.Optimizer(self.time_series[index_val], x_grad, iter_val)                    
        return x_state_new                                       #return the new state
        
    def initialize_time_series(self, maxiter, save_time_series):
        if (save_time_series is True):
            self.time_series = np.zeros((maxiter+self.max_L+1,self.n))
        else:
            self.time_series = np.zeros((self.max_L+1,self.n))
            
    def compute_index_val(self, save_time_series, i):
        if (save_time_series is True):
            return i+self.max_L   
        return self.max_L
         
    def add_new_state(self, save_time_series, x_state_new,i):
        if (save_time_series is True):
            self.time_series[i+1+self.max_L,:] = x_state_new   #add the updated values to the time series
            x_state_old = self.time_series[i+self.max_L,:]
        else:
            self.time_series = np.roll(self.time_series,-1,axis=0)
            self.time_series[-1,:] = x_state_new
            x_state_old = self.time_series[-2,:]    
        return x_state_old 
        
    def compute_time_series(self, tol=1e-10, maxiter=5000, use_delays=False, random=True, symmetric_delays=False,
                            D=None, shrink=False, save_time_series=True):
        """computes the time series using the passed Optimizer from __init__, saves convergence
           and time_series (if specified) which is an array of the states
           
           Parameters - 
               tol (float) - the tolerance of convergence before ending the optimization
               maxiter (int) - the max number of iterations before determining it did not converge
               use_delays (bool) - whether or not to call the use_delay method which adds delays to
               the state vector
        """  
        conv_bool = False                                       #initialize the convergence boolean  
        self.initialize_time_series(maxiter,save_time_series)
        self.num_max_delay = self.max_L                         #initialize number for max delay of iteration
        if (self.Optimizer.initialized is False):
            self.Optimizer.initialize(self.x_init)
        self.add_copies()                 #add copies to the time series for the delay
        x_state_new = self.x_init         #initialize new state array/matrix
        for i in range(maxiter):          #start optimizer iterations      
            index_val = self.compute_index_val(save_time_series,i)       #compute the index selection value
            if (use_delays is True):
                new_value = self.use_delay(index_val = index_val, random=random, 
                                           D=D, symmetric_delays=symmetric_delays, 
                                           shrink=shrink, iter_val=i+1)  #use_delay to get state
                if (new_value is None):
                    break
                x_state_new = new_value
            else:            
                value = self.time_series[index_val]    #computation without delays
                x_grad = self.grad(value)
                #handle the exception case in the gradient problem
                if (x_grad is None):
                    break
                x_state_new = self.Optimizer(self.time_series[index_val], x_grad, i+1)  #compute the update step
            x_state_old = self.add_new_state(save_time_series, x_state_new,i) #add the state to the time series
            comp_val = np.linalg.norm(x_state_new - x_state_old)
            #track losses over time (temporal complexity dependent on computation cost of functional value)
            if self.logging is True:
                loss_val = self.loss_function(x_state_new)
                self.loss_list.append(loss_val)
                print("Iteration: {}, Loss: {}, Distance from Previous State: {}".format(i, loss_val, comp_val))
            if (comp_val < tol):
                conv_bool = True
                break
        #save algorithm variables        
        self.Optimizer.initialized = False                    #reset the input optimizer
        if (save_time_series is True):
            self.time_series = self.time_series[self.max_L:i+2+self.max_L,:] #remove copies and end zeros
        self.final_state = x_state_new                        #save the final state
        self.final_val = self.loss_function(self.final_state) #save the final loss value
        self.conv = conv_bool                                 #save convergence boolean    
            
            
