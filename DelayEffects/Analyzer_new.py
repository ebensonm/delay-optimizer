# Analyzer_new.py 

import numpy as np
import itertools
from matplotlib import pyplot as plt
from matplotlib import ticker
import matplotlib as mpl
import pandas as pd
import warnings
from Optimizer_Scripts.learning_rate_generator import generate_learning_rates
from Optimizer_Scripts import Delayer
from Optimizer_Scripts import optimizers 
from Optimizer_Scripts import functions


class Analyzer:
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
               tol(float): the tolerance value used in the system
               compute_loss(bool): whether the Delayer class should compute the
                                   loss values
               save_grad(bool): whether the Delayer class should save gradient 
                                values
        """
        # Attributes for the Analyzer class
        self.n = n
        self.loss_name = loss_name
        self.optimizer_name = optimizer_name
        self.maxiter = maxiter
        self.tol = tol
        self.initialize_functions()
        self.initialize_params()
        
        # Initialize the lists of values to save
        self.x_inits = None
        self.grid = None
        self.final_states = list()
        self.states = list()
        self.final_losses = list()
        self.losses = list()
        self.final_grads = list()
        self.grads = list()
        self.iters = list()
        self.conv = list()
        
        
    def initialize_functions(self):
        """Initializes the loss and gradient functions based on loss_name. 
        Also initilizes the default range grid and the known minimizer 
        for the function.
        """
        if self.loss_name == 'Rosenbrock':
            self.loss = functions.rosenbrock_gen(self.n)
            self.grad = functions.rosen_deriv_gen(self.n)
            self.range_grid = [-10.,10.]
            self.minimizer = np.ones(self.n)
        elif self.loss_name == 'Zakharov':
            self.loss = functions.zakharov_gen(self.n)
            self.grad = functions.zakharov_deriv_gen(self.n)
            self.range_grid = [-10.,10.]
            self.minimizer = np.zeros(self.n)
        elif self.loss_name == 'Ackley':
            self.loss = functions.ackley_gen(self.n)
            self.grad = functions.ackley_deriv_gen(self.n)
            self.range_grid = [-32.,32.]
            self.minimizer = np.zeros(self.n)
        elif self.loss_name == 'Rastrigin':
            self.loss = functions.rastrigin_gen(self.n)
            self.grad = functions.rast_deriv_gen(self.n)
            self.range_grid = [-32.,32.]
            self.minimizer = np.zeros(self.n)
        else:
            raise ValueError("The '{}' function has not been implemented."\
                             .format(self.loss_name))
            
            
    def get_params(self, delay_type, param_type='optimal', filename='../final_params.csv'):
        # Get the data we need and filter by function
        params = pd.read_csv(filename)
        params = params[params.loss_name == self.loss_name]
        params = params[params.dim == self.n]
        
        if param_type=='default':
            return {'max_learning_rate': 2.98, 'min_learning_rate': 0.23, 
                    'step_size': 740.}
        
        elif param_type=='undelayed' or delay_type.name=='undelayed':
            params = params[params.use_delays == False]
            params = params.drop(columns=['loss_name','dim','max_L','delay_type',
                                          'use_delays'])
            params = params.to_dict('index').values()
            
        elif param_type=='optimal':
            # Get specified delayed parameters
            params = params[params.use_delays == True]
            params = params[params.max_L == delay_type.max_L]
            params = params[params.delay_type == delay_type.name]  # Might need support for more delay_type vars
            params = params.drop(columns=['loss_name','dim','max_L','delay_type',
                                          'use_delays'])
            params = params.to_dict('index').values()
            
        else:
            raise ValueError("Invalid parameter type.")
        
        if len(list(params)) > 0 :
            return list(params)[0]
        else:
            warnings.warn("No optimal hyperparameters found. Using "
                          "default hyperparameters.")
            return self.get_params(delay_type, param_type='default')


    def calculate_save_values(self, delay_type, break_opt=True, save_state=True, 
                              save_loss=True, save_grad=False, save_iters=True, 
                              **kwargs):
        """Run the optimization on the initial points already initialized and 
        saves values to be plotted.
        
        Parameters:
            delay_type: DelayType instance
            break_opt(bool): whether optimization should stop when convergence 
                             criteria is met
            save_state(bool): whether the class should save the time series of 
                              each point
            save_loss(bool): whether the class should save loss values over 
                             time for each point
            save_grad(bool): whether the class should save gradient values over
                             time for each point
            save_iters(bool): whether the class should save the number of 
                              iterations each point takes to converge
        """
        # Initialize
        self.delete_data()
        self.delay_type = delay_type
                
        # Initialize
        lr_params = self.get_params(delay_type)   # TODO: Function that gets best params by function / delay type
        self.initialize_optimizer(lr_params)
        self.initialize_delayer(delay_type, compute_loss=save_loss, 
                                save_grad=save_grad)
            
        for x_init in self.x_inits:
            # Perform the optimization for each initial point 
            x_init = np.asarray(x_init)
            self.run_single_start(x_init, delay_type, break_opt)
                
            # Save values
            if save_state:
                self.time_series.append(self.delayer.time_series)
                self.final_states.append(self.delayer.final_state)
            if save_loss:
                self.loss_vals.append(self.delayer.loss_list)
                self.final_losses.append(self.delayer.final_val)
            if save_grad:
                self.grad_vals.append(self.delayer.grad_list)
                self.final_grads.append(self.delayer.grad_list[-1])
            if save_iters:
                self.iters.append(len(self.delayer.time_series))
            self.conv.append(self.delayer.conv)
                    
            # Recreate delayer and optimizer to reset the lr generator
            del self.delayer
            del self.optimizer
            self.initialize_optimizer(lr_params)
            self.initialize_delayer(delay_type, compute_loss=save_loss, 
                                    save_grad=save_grad)
        del self.delayer
        del self.optimizer
            
        self.add_initial_vals(save_loss, save_grad) 
        
    
        
