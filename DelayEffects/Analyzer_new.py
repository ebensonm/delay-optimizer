# Analyzer_new.py 

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
import matplotlib as mpl
import pandas as pd
import warnings
from Optimizer_Scripts.learning_rate_generator import generate_learning_rates
from Optimizer_Scripts.Delayer_new import Delayer
from Optimizer_Scripts import optimizers 
from Optimizer_Scripts import functions
import sys


class FuncOpt:
    """Class to analyze the effects of delayed optimization on various 
    functions. Includes functions to calculate the time series, loss values, 
    and gradient values over time, and to graph plots of that data.
    """
    def __init__(self, loss_name, n, optimizer_name='Adam', maxiter=2000, 
                 tol=1e-5, **kwargs):
        """The initializer for the Analyzer class.
            
           Parameters: 
               loss_name(str): name of the loss function to be analyzed
               n(int): the dimension of the state vector
               optimizer_name(str): name of the optimization algorithm to be 
                                    used
               maxiter(int): the maximum number of iterations in the system
               tol(float): the convergence tolerance used in optimization
        """
        # Attributes for the Analyzer class
        self.loss_name = loss_name
        self.n = n
        self.optimizer_name = optimizer_name
        self.maxiter = maxiter
        self.tol = tol
        self.initialize_function()
        
        # Initialize the lists of values to save
        self.x_inits = []
        self.grid = None
        self.states = []
        self.loss_vals = []
        self.grad_vals = []
        self.iters = []
        self.conv = []
        
        
    def initialize_function(self):
        """Initializes the loss and gradient functions based on loss_name. 
        Initilizes the default domain and the known minimizer for the function.
        """
        if self.loss_name == 'Rosenbrock':
            self.loss = functions.rosenbrock_gen(self.n)
            self.grad = functions.rosen_deriv_gen(self.n)
            self.domain = [-10.,10.]
            self.minimizer = np.ones(self.n)
        elif self.loss_name == 'Zakharov':
            self.loss = functions.zakharov_gen(self.n)
            self.grad = functions.zakharov_deriv_gen(self.n)
            self.domain = [-10.,10.]
            self.minimizer = np.zeros(self.n)
        elif self.loss_name == 'Ackley':
            self.loss = functions.ackley_gen(self.n)
            self.grad = functions.ackley_deriv_gen(self.n)
            self.domain = [-32.,32.]
            self.minimizer = np.zeros(self.n)
        elif self.loss_name == 'Rastrigin':
            self.loss = functions.rastrigin_gen(self.n)
            self.grad = functions.rast_deriv_gen(self.n)
            self.domain = [-32.,32.]
            self.minimizer = np.zeros(self.n)
        else:
            raise ValueError("The '{}' function has not been implemented."\
                             .format(self.loss_name))
                
                
    def initialize_points(self, sample, num_points=1):
        """Initialize the initial points for the optimization.
        
        Parameters:
            sample(str): 'random' to choose random initial points
                         'grid' to create an evenly spaced grid of num_points^2 
                             points
                         (ndarray) to use the given array of points
            points(list): list of points to use for sample='given'
        """
        self.reset()        # Reset previous points, delete saved data
        
        if type(sample) == np.ndarray:  # Given array of points
            sample = np.asarray(sample)
            if sample.shape[1] != self.n:
                raise ValueError("Array shape does not align with function "
                                 "dimension.")
            self.x_inits = sample
        elif sample == 'random':        # Random points in domain
            self.x_inits = np.random.uniform(self.domain[0], self.domain[1], 
                                             size=(num_points,self.n))
        elif sample == 'grid':          # Grid of points in domain
            if self.n != 2:
                raise ValueError("Grid sample type requires a function "
                                 "dimension of exactly 2.")
            x = np.linspace(self.domain[0], self.domain[1], num_points)
            self.grid = np.asarray(np.meshgrid(x, x))
            self.x_inits = self.grid.reshape((self.n, num_points**2)).T
        else:
            raise ValueError("The specified sample type does not exist.")
            
                
    def initialize_optimizer(self, lr_params, beta_1=0.9, beta_2=0.999):
        """Initializes the optimizer and its parameters."""
        if self.optimizer_name == 'Adam':
            params = {'beta_1': beta_1, 'beta_2': beta_2}
            params['learning_rate'] = generate_learning_rates(False, lr_params)
            self.optimizer = optimizers.Adam(params)
        else:
            raise ValueError("The '{}' optimizer has not been implemented."\
                             .format(self.optimizer_name))            
                
                
    def initialize_delayer(self, save_state=True, save_loss=True, 
                           save_grad=True):
        """Initialize the Delayer class item for optimization."""
        self.delayer = Delayer(self.n, self.delay_type, self.loss, self.grad, 
                               self.optimizer, save_state, save_loss, 
                               save_grad)            
                
                
    def get_params(self, delay_type, param_type='optimal', 
                   filename='../final_params.csv', **kwargs):
        # Get the data we need and filter by function
        params = pd.read_csv(filename, index_col=0)
        params = params[params.loss_name == self.loss_name]
        params = params[params.dim == self.n]
        
        if param_type=='default':
            return {'max_learning_rate': 2.98, 'min_learning_rate': 0.23, 
                    'step_size': 740.}
        
        elif param_type=='given':
            return {'max_learning_rate': kwargs['max_learning_rate'], 
                    'min_learning_rate': kwargs['min_learning_rate'], 
                    'step_size': kwargs['step_size']}
        
        elif param_type=='undelayed' or delay_type.name=='undelayed':
            params = params[params.use_delays == False]
            params = params.drop(columns=['loss_name','dim','max_L',
                                          'delay_type','use_delays'])
            params = params.to_dict('index').values()
            
        elif param_type=='optimal':
            # Get specified delayed parameters
            params = params[params.use_delays == True]
            params = params[params.max_L == delay_type.max_L]
            params = params[params.delay_type == delay_type.name]  
            params = params.drop(columns=['loss_name','dim','max_L',
                                          'delay_type','use_delays'])
            params = params.to_dict('index').values()
            
        else:
            raise ValueError("Invalid parameter type.")
        
        if len(list(params)) > 0 :
            return list(params)[0]
        else:
            warnings.warn("No optimal hyperparameters found. Using "
                          "default hyperparameters.")
            return self.get_params(delay_type, param_type='default')


    def optimize(self, delay_type, param_type='optimal',  break_opt=True, 
                 save_state=True, save_loss=True, save_grad=False, 
                 save_iters=True, param_file='../final_params.csv',
                 **kwargs):
        """Run the optimization on the initial points already initialized and 
        saves values to be plotted.
        
        Parameters:
            delay_type(DelayType): class object containing delay parameters
            break_opt(bool): whether optimization should stop when convergence 
                             criteria is met
            save_state(bool): whether to save state values over time
            save_loss(bool): whether to save loss values over time 
            save_grad(bool): whether to save gradient values over time 
            save_iters(bool): whether to save the number of iterations taken 
        """
        # Check if points have been initialized
        if len(self.x_inits) == 0:
            warnings.warn("No points to optimize over have been initialized.")
            return 1
        
        # Initialize
        self.delete_data()
        self.delay_type = delay_type
        self.lr_params = self.get_params(delay_type, param_type, param_file, 
                                         **kwargs)   
        self.initialize_optimizer(self.lr_params)
        self.initialize_delayer(save_state, save_loss, save_grad)
            
        for x_init in self.x_inits:
            # Perform the optimization for each initial point 
            self.delayer.optimize(x_init, self.maxiter, self.tol, break_opt)
            
            # Save values
            if save_state is True:
                self.states.append(np.asarray(self.delayer.state_list))
            if save_loss is True:
                self.loss_vals.append(self.delayer.loss_list)
            if save_grad is True:
                self.grad_vals.append(self.delayer.grad_list)
            if save_iters is True:
                self.iters.append(self.delayer.iters)
            self.conv.append(self.delayer.conv)
                    
            # Recreate delayer and optimizer to reset the lr generator
            del self.delayer
            del self.optimizer 
            self.initialize_optimizer(self.lr_params)
            self.initialize_delayer(save_state, save_loss, save_grad)
            
        del self.delayer
        del self.optimizer
        self.states = np.asarray(self.states, dtype=object)
        self.loss_vals = np.asarray(self.loss_vals, dtype=object)
        self.grad_vals = np.asarray(self.grad_vals, dtype=object)
        
        
    def get_finals(self, value_list):
        """Returns an array of final values from the given list of sequences"""
        return np.asarray([val[-1] for val in value_list])
      
      
    def get_mean_final(self, value_list):
        """Returns the final mean value of the given list of sequences"""
        return np.mean(self.get_finals(value_list), axis=0)
        
        
    def get_slice(self, dim_tuple):
        """Returns the desired slice of the state data. 
        
        Parameters -
            dim_tuple (tuple): The dimensions to extract
        Returns -
            (ndarray(list(ndarray))): Ragged nested array of sliced state 
                sequences
        """
        return np.array([np.array([it[np.r_[dim_tuple]] for it in point]) 
                         for point in self.states], dtype=object)


    def save(self, filename, dim_tuple):
        delay_dict = vars(self.delay_type).copy()
        delay_type = delay_dict['name']
        delay_dict.pop('name')
        
        np.savez_compressed(filename, loss_name=self.loss_name, n=self.n,
            domain=self.domain, optimizer_name=self.optimizer_name, 
            maxiter=self.maxiter, tol=self.tol, lr_params=self.lr_params,
            states=self.get_slice(dim_tuple), loss_vals=self.loss_vals, 
            grad_vals=self.grad_vals, iters=self.iters, conv=self.conv,
            x_inits=self.x_inits[:,dim_tuple], delay_type=delay_type, 
            **delay_dict,)
        
        
    def load(self, filename):
        obj_data = np.load(filename, allow_pickle=True)     # Load data
        self.reset()                                        # Reset values
        
        if obj_data['loss_name'] != self.loss_name or obj_data['n'] != self.n:
            raise ValueError("Functions from FuncOpt instance and file data "
                             "do not match. Data was not loaded.")
        elif not np.allclose(obj_data['domain'], self.domain):
            warnings.warn("Function domain was updated to match data.")
        elif (obj_data['optimizer_name'] != self.optimizer_name or 
              obj_data['tol'] != self.tol or 
              obj_data['maxiter'] != self.maxiter):
            warnings.warn("Optimizer parameters were updated to match data.")
            
        self.__dict__.update(obj_data)      # Add the data to FuncOpt object
    
    
    def delete_initials(self):
        """Deletes all initialized points"""
        self.x_inits = []
        self.grid = None
            
        
    def delete_states(self):
        """Deletes computed state list"""
        self.states = []
        
        
    def delete_losses(self):
        """Deletes computed loss list"""
        self.loss_vals = []
        
        
    def delete_grads(self):
        """Deletes computed gradient list"""
        self.grad_vals = []
        
        
    def delete_data(self):
        """Deletes all computed values but retains initial values"""
        self.delete_states()
        self.delete_losses()
        self.delete_grads()
        self.iters = []
        self.conv = []
        self.delay_type = None
        self.lr_params = None
        
        
    def reset(self):
        """Resets all values"""
        self.delete_initials()
        self.delete_data()
        
    
        
