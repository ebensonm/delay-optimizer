# Analyzer.py 

import numpy as np
import pandas as pd
import pickle
import bz2
import _pickle as cPickle
import itertools
from matplotlib import pyplot as plt
from matplotlib import ticker
import matplotlib as mpl
from Optimizer_Scripts.learning_rate_generator import generate_learning_rates
from Optimizer_Scripts import Delayer
from Optimizer_Scripts import optimizers 
from Optimizer_Scripts import functions

class Analyzer:
    """Class to analyze the effects of delayed optimization on various 
    functions. Includes functions to calculate the time series, loss values, 
    and gradient values over time, and to graph plots of that data.
    """
    def __init__(self, n, loss_name, optimizer_name='Adam', const_lr=False, 
                 max_L=1, num_delays=1000, maxiter=2000, tol=1e-5, **kwargs):
        """The initializer for the Helper class.
            
           Parameters: 
               n(int): the dimension of the state vector
               loss_name(str): name of the loss function to be analyzed
               optimizer_name(str): name of the optimization algorithm to be 
                                    used
               const_lr (bool): whether to use a constant learning rate
               max_L(int): the maximum delay of the system
               num_delays(int): the number of delays to compute with the system
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
        self.const_lr = const_lr
        self.max_L = max_L
        self.num_delays = num_delays
        self.maxiter = maxiter
        self.tol = tol
        self.initialize_functions()
        self.initialize_params()
        
        # Initialize the lists of values to save
        self.x_inits = None
        self.grid = None
        self.final_states = list()
        self.time_series = list()
        self.final_losses = list()
        self.loss_vals = list()
        self.final_grads = list()
        self.grad_vals = list()
        self.iters = list()
        self.conv = list()
        self.del_final_states = list()
        self.del_time_series = list()
        self.del_final_losses = list()
        self.del_loss_vals = list()
        self.del_final_grads = list()
        self.del_grad_vals = list()
        self.del_iters = list()
        self.del_conv = list()
            
        
    def initialize_functions(self):
        """Initializes the loss and gradient functions based on loss_name. 
        Also initilizes the default range grid and the known minimizer for the 
        function.
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
        
        
    def initialize_params(self):
        """Set the delayed and undelayed optimal parameters closest to those 
        computed.
        """
        # Get the data we need
        hyperparams = pd.read_csv('../final_params.csv', index_col=0)
        params = hyperparams[hyperparams["loss_name"] == self.loss_name]
        params = params[params["dim"] == self.n]
        
        # Separate and organize the values we need
        params = params.drop(columns=['loss_name','dim','max_L','delay_type'])
        del_params = params[params.use_delays == True]
        undel_params = params[params.use_delays == False]
        
        # Create parameter dictionaries
        del_params = del_params.drop(columns=['use_delays'])\
            .to_dict('index').values()
        undel_params = undel_params.drop(columns=['use_delays'])\
            .to_dict('index').values()
        
        # Set undelayed params
        if len(list(undel_params)) == 0:
            self.params = {'max_learning_rate': 2.98,
                           'min_learning_rate': 0.23,
                           'step_size': 740.}
        else:
            self.params = list(undel_params)[0]
            
        # Set delayed params
        if len(list(del_params)) == 0:
            self.del_params = {'max_learning_rate': 2.98,
                               'min_learning_rate': 0.23,
                               'step_size': 740.}
        else:
            self.del_params = list(del_params)[0]
        
        
    def initialize_optimizer(self, delayed, beta_1=0.9, beta_2=0.999):
        """Initializes the optimial paramters and the optimizer."""
        params = dict()
        params['beta_1'] = beta_1
        params['beta_2'] = beta_2
    
        if delayed is True:
            params['learning_rate'] = generate_learning_rates(self.const_lr, 
                                                              self.del_params)
        else:
            params['learning_rate'] = generate_learning_rates(self.const_lr, 
                                                              self.params)
        if self.optimizer_name == 'Adam':
            self.optimizer = optimizers.Adam(params)
        else:
            raise ValueError("The '{}' optimizer has not been implemented."\
                             .format(self.optimizer_name))
            
            
    def initialize_delayer(self, compute_loss=True, save_grad=True):
        """Initialize the Delayer class item for optimization."""
        self.delayer = Delayer.Delayer(self.n, self.optimizer, self.loss, 
                                       self.grad, max_L=self.max_L, 
                                       num_delays=self.num_delays, 
                                       compute_loss=compute_loss, 
                                       save_grad=save_grad)
            
            
    def initialize_points(self, num_points, sample, points=None):
        """Initialize the initial points for the optimization.
        
        Parameters:
            num_points(int): number of points to initialize
            sample(str): 'random' to choose random initial points
                         'same' to have all be the same randomly chosen point 
                         'grid' to create an evenly space grid of num_points^2 
                             points
                         'given' to use the given points
            points(list): list of points to use for sample='given'
        """
        # Reset previous points, delete saved data
        self.clear()
        
        # Set the points on which to optimize
        if sample == 'random':
            self.x_inits = np.random.uniform(self.range_grid[0], 
                                             self.range_grid[1], 
                                             size=(num_points,self.n))
        elif sample == 'grid':
            self.x_inits, self.grid = self.create_grid(num_points)
        elif sample == 'same':
            xint = np.random.uniform(self.range_grid[0], self.range_grid[1], 
                                     size=self.n)
            self.x_inits = np.tile(xint, (num_points, 1))
        elif sample == 'given':
            if points is None:
                raise ValueError("Must provide points for type_test='given'.")
            self.x_inits = np.asarray(points)
        else:
            raise ValueError("Test type '{}' does not exist.".format(sample))
    
    
    def create_grid(self, num_points):
        """Helper function used to initialize an evenly spaced grid of initial 
        points.
        
        Parameters:
            num_points(int): number of points to use for each dimension of the
                             grid (total number of points is num_points^n)
        Returns:
            x_inits(ndarray): an array representing the initial points of the 
                              grid
            grid(meshgrid): a meshgrid object used for plotting contour plots
        """
        x = np.linspace(self.range_grid[0], self.range_grid[1], num_points)
        list_xs = [x for i in range(self.n)]
        initials = itertools.product(*list_xs)
        
        # Format itertools product as an array
        x_inits = list()
        for point in initials:
            x_inits.append(point)
        x_inits = np.asarray(x_inits)
        
        X, Y = np.meshgrid(x, x)
        grid = [X, Y]
        
        return x_inits, grid
    
            
    def initialize_vars(self, **kwargs):
        """Updates parameters by keyword argument."""
        if 'range_grid' in kwargs:
            self.range_grid = kwargs['range_grid']
        if 'max_L' in kwargs:
            self.max_L = kwargs['max_L']
        if 'num_delays' in kwargs:
            self.num_delays = kwargs['num_delays']
        if 'maxiter' in kwargs:
            self.maxiter = kwargs['maxiter']
        if 'tol' in kwargs:
            self.tol = kwargs['tol']
            
            
    def run_single_start(self, x_init, use_delays, D=None, random=True, 
                         break_opt=True):
        """Runs the optimization for a single point."""
        self.delayer.x_init = x_init
        self.delayer.compute_time_series(tol=self.tol, maxiter=self.maxiter, 
                                         use_delays=use_delays, random=random, 
                                         D=D, break_opt=break_opt)
            
            
    def calculate_save_values(self, delayed, D=None, random=True, 
                              break_opt=True, save_state=True, save_loss=True, 
                              save_grad=True, save_iters=True, **kwargs):
        """Run the optimization on the initial points already initialized and 
        saves values to be plotted.
        
        Parameters:
            delayed: whether to delay the optimization or not ('both' 
                     calculates both)
            max_L(int): the maximum delay of the system
            num_delays(int): the number of delays to compute with the system
            maxiter(int): the maximum number of iterations in the system
            tol(float): the tolerance value used in the system
            D(list): list of deay distributions for the Delayer class
            random(bool): whether delays should be stochastic or not
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
        if delayed == 'both':
            # If 'both' run the function for both True and False
            self.calculate_save_values(False, D, random, break_opt, save_state, 
                                       save_loss, save_grad, save_iters, 
                                       **kwargs)
            self.calculate_save_values(True, D, random, break_opt, save_state, 
                                       save_loss, save_grad, save_iters, 
                                       **kwargs)
        elif delayed in (True, False):
            # Reset old data
            if delayed is True:
                self.delete_delayed_data()
            else:
                self.delete_undelayed_data()
                
            # Initialize
            self.initialize_vars(**kwargs)
            self.initialize_optimizer(delayed)
            self.initialize_delayer(compute_loss=save_loss,
                                    save_grad=save_grad)
            
            for x_init in self.x_inits:
                # Perform the optimization for each initial point 
                x_init = np.asarray(x_init)
                self.run_single_start(x_init, delayed, D, random, break_opt)
                
                # Save values
                if delayed is True:
                    if save_state:
                        self.del_time_series.append(self.delayer.time_series)
                        self.del_final_states.append(self.delayer.final_state)
                    if save_loss:
                        self.del_loss_vals.append(self.delayer.loss_list)
                        self.del_final_losses.append(self.delayer.final_val)
                    if save_grad:
                        self.del_grad_vals.append(self.delayer.grad_list)
                        self.del_final_grads.append(self.delayer.grad_list[-1])
                    if save_iters:
                        self.del_iters.append(len(self.delayer.time_series))
                    self.del_conv.append(self.delayer.conv)
                else:
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
                self.initialize_optimizer(delayed)
                self.initialize_delayer()
            del self.delayer
            del self.optimizer
            
            self.add_initial_vals(delayed, save_loss, save_grad) 
            
        else:   # Check for incorrect inputs
            raise ValueError("Variable 'delays' must be 'both', True, or "
                             "False.")   
        
        return
            
         
    def add_initial_vals(self, delayed, save_loss=True, save_grad=False):
        """The Delayer class does not include the loss or gradient values for 
        the initial points in the time series. This function adds those values 
        to the respective class attributes.
        """
        if not (save_loss or save_grad):
            return
        else:
            for i in range(len(self.x_inits)):
                if save_loss:
                    loss = self.loss(self.x_inits[i])
                    if delayed is True:
                        self.del_loss_vals[i].insert(0, loss)
                    if delayed is False:
                        self.loss_vals[i].insert(0, loss)
                if save_grad:
                    grad = self.grad(self.x_inits[i])
                    if delayed is True:
                        self.del_grad_vals[i].insert(0, grad)
                    if delayed is False:
                        self.grad_vals[i].insert(0, grad)
            
         
    def extract_values(self, delayed, focus):
        """Uses the delayed boolean and focus string to return the desired 
        data.
        
        Parameters:
            delayed(bool): whether the desired data is for delayed or undelayed
            focus(str): 'loss', 'grad', 'state', or 'iters' values to return
            
        Returns:
            (list(list)): requested values over time for all points
            (list): final values for all points
        """
        if delayed is True:
            if focus == 'state':
                return self.del_time_series, self.del_final_states
            if focus == 'loss':
                return self.del_loss_vals, self.del_final_losses
            if focus == 'grad':
                return self.del_grad_vals, self.del_final_grads
            if focus == 'iters':
                return None, self.del_iters
        if delayed is False:
            if focus == 'state':
                return self.time_series, self.final_states
            if focus == 'loss':
                return self.loss_vals, self.final_losses
            if focus == 'grad':
                return self.grad_vals, self.final_grads
            if focus == 'iters':
                return None, self.iters
            

    def set_bins(self, num_bins, values, del_values):
        """Sets the bin size equally for delayed and undelayed final values"""
        min_val = min(min(values), min(del_values))
        max_val = max(max(values), max(del_values))
        return np.linspace(min_val, max_val, num_bins)
        
            
    def extract_dims(self, point_ind, dim_tuple, delayed):
        """Pulls the desired data from the saved values."""
        data = list()
        if delayed is True:
            data.append(self.del_time_series[point_ind][:,dim_tuple[0]]\
                        .tolist())
            data.append(self.del_time_series[point_ind][:,dim_tuple[1]]\
                        .tolist())
        else: 
            data.append(self.time_series[point_ind][:,dim_tuple[0]].tolist())
            data.append(self.time_series[point_ind][:,dim_tuple[1]].tolist())
        return np.array(data)
        
    
    def ravel_data(self, values):
        """Extracts the nested list of values into a single list."""
        new_list = list()
        for data in values:
            for element in data:
                new_list.append(element)
        return new_list
    
    
    def plot_colorbar(self, fig, axis, image):
        colorbar = fig.colorbar(image, ax=axis)
        colorbar.set_alpha(1)
        colorbar.draw_all()
        
        
    def initialize_plot(self, **kwargs):
        """Returns the figure and axis to plot on. Generates these if they are
        not given in the keyword arguments.
        """
        if 'ax' in kwargs:
            ax = kwargs['ax']
            plt.sca(ax)
            fig = plt.gcf()
            del kwargs['ax']
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10,10))
        
        return fig, ax, kwargs
    
        
    def plot_finals(self, delayed, focus='loss', **kwargs):
        """Plot a histogram of the previously computed final values.
        
        Parameters:
            delayed (bool/str): Values to plot (True, False, or 'both')
            focus (str): Values to plot ('loss' or 'grad')
        """
        # Error checker
        if delayed not in ('both', True, False):
            raise ValueError(r"'delayed' argument must be 'both', True, or "
                             "False not '{}'.".format(delayed))
        if focus not in ('loss', 'grad', 'state', 'iters'):
            raise ValueError(r"Finals plot focus must be 'loss', 'grad', "
                             "'state', or 'iters', not '{}'.".format(focus))
        
        # Initialize plot
        fig, ax, kwargs = self.initialize_plot(**kwargs)
            
        if delayed == 'both': 
            # Set default values if not specified
            default = {'alpha':0.3, 'bins':25}
            kwargs = {**default, **kwargs}
            
            # Set fixed bins between delayed and undelayed values
            if type(kwargs['bins']) == int:
                final_vals = self.extract_values(False, focus)[1]
                del_final_vals = self.extract_values(True, focus)[1]
                if focus == 'state':
                    final_vals = self.ravel_data(final_vals)
                    del_final_vals = self.ravel_data(del_final_vals)
                kwargs['bins'] = self.set_bins(kwargs['bins'], final_vals, 
                                               del_final_vals)
            
            ax = self.plot_finals(False, focus, ax=ax, color='b', **kwargs)
            ax = self.plot_finals(True, focus, ax=ax, color='r', **kwargs)
                
        else:
            # Set default values if not specified
            default = {'alpha':0.8, 'bins':25, 'color':'b'}
            kwargs = {**default, **kwargs}
            
            # Get data to plot
            final_vals = self.extract_values(delayed, focus)[1]
            if focus == 'state':
                final_vals = self.ravel_data(final_vals)
                
            # Plot the histogram of the data
            ax.hist(final_vals, **kwargs)
            
        return ax
        
            
    def plot_iters(self, delayed, focus='iters', plot_dims=(0,1), points=None, 
                   iters=None, colorbar=False, **kwargs):
        """Plot the scatterplot of state values for each iteration of 
        optimization previously computed
        
        Parameters:
            delayed (bool/str): Values to plot (True, False, or 'both')
            focus (str): Values to plot ('loss' or 'grad')
            plot_dims (tuple): Two dimensions to plot against each other
        """
        # Error checker
        if delayed not in ('both', True, False):
            raise ValueError(r"'delayed' argument must be 'both', True, or "
                             "False not '{}'.".format(delayed))
        if focus not in ('loss', 'grad', 'iters'):
            raise ValueError(r"Iters plot focus must be 'loss', 'grad', or "
                             "'iters' not '{}'.".format(focus))
            
        # Initialize plot
        fig, ax, kwargs = self.initialize_plot(**kwargs)
            
        if delayed == 'both':
            ax = self.plot_iters(False, focus, plot_dims, ax=ax, **kwargs)
            ax = self.plot_iters(True, focus, plot_dims, ax=ax, cmap='autumn',
                                 **kwargs)

        else:
            # Set default parameters
            default = {'alpha':0.01, 's':20, 'cmap':'winter_r'}
            
            # By default plot all points for all iterations
            if points is None:
                points = np.arange(len(self.x_inits))
            if iters is None:
                iters = self.maxiter
            
            # Determine scaling for plotting
            if focus == 'iters':
                final_iters = self.extract_values(delayed, 'iters')[1]
                values = [np.arange(it) for it in final_iters]
                vmax, vmin = self.maxiter, 1
            else:
                values = self.extract_values(delayed, focus)[0]
                vmax = max([np.max(point) for point in values])
                vmin = self.tol
                
            if 'norm' in kwargs and kwargs['norm'] == 'log':
                kwargs['norm'] = mpl.colors.LogNorm(vmin, vmax)
            else:
                kwargs['norm'] = mpl.colors.Normalize(vmin, vmax)
            kwargs = {**default, **kwargs}
            
            # Plot values
            for i in points:
                data = self.extract_dims(i, plot_dims, delayed)
                if abs(iters) < data.shape[1]:
                    if iters < 0:
                        data = data[:,iters:]
                        colors = values[i][iters:]
                    else:
                        data = data[:,:iters]
                        colors = values[i][:iters]
                else:
                    colors = values[i]
                im = ax.scatter(data[0], data[1], c=colors, **kwargs)
                
            # Format axis
            if colorbar is True:
                self.plot_colorbar(fig, ax, im)
            ax.set_xlim(self.range_grid)
            ax.set_ylim(self.range_grid)
            
        return plt.gca()
        
        
    def plot_paths(self, delayed, plot_dims=(0,1), points=None, iters=None, 
                   **kwargs):
        """Plot the path of the previously computed optimization over time.
        
        Parameters:
            delayed (bool/str): Values to plot (True, False, or 'both')
            time_plot (list): Indices of the paths to plot
            plot_dims (tuple): Two dimensions to plot against each other
        """
        # Error checker
        if delayed not in ('both', True, False):
            raise ValueError(r"'delayed' argument must be 'both', True, or "
                             "False not '{}'.".format(delayed))
        
        # Initialize plot
        fig, ax, kwargs = self.initialize_plot(**kwargs)
            
        if delayed == 'both':
            ax = self.plot_paths(False, plot_dims, ax=ax, **kwargs)
            ax = self.plot_paths(True, plot_dims, ax=ax, **kwargs)
            
        else:
            # Set default values if not specified
            default = {'alpha':0.5, 'lw':3, 'c':'k'}
            kwargs = {**default, **kwargs}
            
            # By default plot all points for all iterations
            if points is None:
                points = np.arange(len(self.x_inits))
            if iters is None:
                iters = self.maxiter
            
            # Plot values
            for i in points:
                data = self.extract_dims(i, plot_dims, delayed)
                if abs(iters) < data.shape[1]:
                    if iters < 0:
                        data = data[:,iters:]
                    else:
                        data = data[:,:iters]
                ax.plot(data[0], data[1], **kwargs)
                
                
            # Format axis
            ax.set_xlim(self.range_grid)
            ax.set_ylim(self.range_grid)
            
        return ax
        
        
    def plot_basin(self, delayed, focus='loss', colorbar=False, **kwargs):
        """Plot a contour plot of the basin of attraction from the previously
        computed optimization.
        
        Parameters:
            delayed (bool): Delayed or undelayed values
            focus (str): Values to plot ('loss' or 'grad')
        """
        # Error checker
        if type(delayed) != bool:
            raise ValueError(r"'delayed' argument must be True or False for "
                             "basin plots, not '{}'.".format(delayed))
        
        if focus not in ('loss', 'grad', 'iters'):
            raise ValueError("Basin plot type must have a focus of 'loss', "
                             "'grad', or 'iters'.")
            
        if self.grid is None:
            raise ValueError("Basin plot type is only compatible with "
                             "'grid' point generation.")
        
        if self.n != 2:
            raise NotImplementedError("Basin plot type is only implemented "
                                      "for functions of dimension 2.")
        
        # Initialize plot
        fig, ax, kwargs = self.initialize_plot(**kwargs) 
        
        # Set when nonconvergent points are dropped
        drop_vals = False
        if focus == 'iters':
            drop_vals = True
        
        # Get values to plot
        X, Y = self.grid
        final_values = self.extract_values(delayed, focus)[1]
        
        # Format data for contour plot
        if drop_vals is True:
            if delayed is True:
                Z = np.where(self.del_conv, final_values, np.nan)
            else:
                Z = np.where(self.conv, final_values, np.nan)
        else:
            Z = final_values
        Z = np.resize(Z, (len(X),len(Y))).T
        
        # Set default values if not specified
        default = {'cmap':'winter_r'}
        kwargs = {**default, **kwargs}
        
        # Plot values
        im = ax.contourf(X, Y, Z, **kwargs)
        
        # Format axis
        ax.patch.set_color('.25')
        ax.set_xlim(self.range_grid)
        ax.set_ylim(self.range_grid)
        
        if colorbar is True:
            self.plot_colorbar(fig, ax, im)
            
        return ax
    
    
    def plot_conv(self, delayed, plot_dims=(0,1), drop_vals=False, **kwargs):
        """Plot the final state values of the optimization
        
        Parameters:
            delayed (bool/str): Values to plot (True, False, or 'both')
            plot_dims (tuple): Two dimensions to plot against each other
            drop_vals (bool): Whether to drop final points that did not converge
        """
        # Error checker
        if delayed not in ('both', True, False):
                raise ValueError(r"'delayed' argument must be 'both', True, or "
                                 "False not '{}'.".format(delayed))
        
        # Initialize plot
        fig, ax, kwargs = self.initialize_plot(**kwargs)
        
        if delayed == 'both':
            self.plot_final_states(False, plot_dims, drop_vals, ax=ax, c='g', 
                                   **kwargs)
            self.plot_final_states(True, plot_dims, drop_vals, ax=ax, c='o', 
                                   **kwargs)
        
        else:
            # Set default parameters if not specified
            default = {'alpha':0.2, 'c':'g', 's':30}
            kwargs = {**default, **kwargs}
            
            # Get values to plot
            values = self.extract_values(delayed, 'state')[1]
            data = np.array([(point[plot_dims[0]], point[plot_dims[1]]) for 
                             point in values])
            
            # Drop points that did not converge if desired
            if drop_vals is True:
                if delayed is True:
                    data = data[self.del_conv]
                else:
                    data = data[self.conv]
    
            # Plot
            ax.scatter(data[:,0], data[:,1], **kwargs)
            
            # Format axis
            ax.set_xlim(self.range_grid)
            ax.set_ylim(self.range_grid)
            
        return plt.gca()
    
    
    def plot_contour(self, num_points=250, **kwargs):
        """Create a contour plot of the function."""
        # Initialize plot
        fig, ax, kwargs = self.initialize_plot(**kwargs)
        
        # Create grid of points
        points, grid = self.create_grid(num_points)
        X, Y = grid
        
        # Get loss values over the grid of points
        Z = np.array([self.loss(point) for point in points])
        Z = np.resize(Z, (len(X),len(Y))).T
    
        # Set default values if not specified
        default = {'linewidths':3, 'cmap':'autumn', 
                   'locator':ticker.LogLocator()}
        kwargs = {**default, **kwargs}
        
        ax.contour(X, Y, Z, **kwargs)
        
        return ax
    
    
    def plot_inits(self, plot_dims=(0,1), points=None, **kwargs):
        """Plot the initial points."""
        # Initialize plot
        fig, ax, kwargs = self.initialize_plot(**kwargs)
        
        # Set default parameters if not specified
        default = {'alpha':1., 's':30, 'c':'b'}
        kwargs = {**default, **kwargs}
        if points is None:
            points = np.arange(len(self.x_inits))
            
        # Get values to plot
        data = self.x_inits[points][:,plot_dims]
        
        # Plot
        ax.scatter(data[:,0], data[:,1], **kwargs)
        
        # Format axis
        ax.set_xlim(self.range_grid)
        ax.set_ylim(self.range_grid)
        
        return ax
        
        
    def plot_results(self, plot_type, **kwargs):
        """Wrapper function for plotting function types."""
        # Parse by plot type
        if plot_type == "finals":
            ax = self.plot_finals(**kwargs)
        elif plot_type == "iters":
            ax = self.plot_iters(**kwargs)
        elif plot_type == "paths":
            ax = self.plot_paths(**kwargs)
        elif plot_type == "basin":
            ax = self.plot_basin(**kwargs)
        elif plot_type == "contour":
            ax = self.plot_contour(**kwargs)
        elif plot_type == "conv":
            ax = self.plot_conv(**kwargs)
        elif plot_type == "inits":
            ax = self.plot_inits(**kwargs)
        else:
            raise ValueError(r"Plot 'type' parameter must be \"finals\", "
                             "\"iters\", \"paths\", \"basin\", \"contour\", "
                             "\"conv\", or \"inits\" not {}".format(plot_type))
            
        return ax
        
        
    def plot_list(self, plots, ax=None):
        """Unpacks a list of dictionaries representing plots and plots each 
        on the same axis
        """
        # Initialize plot
        if ax is not None:
            plt.sca(ax)
            fig = plt.gcf()
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10,10))
            
        # Check if 'plots' is actually a single plot 
        if type(plots) == dict:
            plots = [plots]
        
        # Iterate through each plot in the list
        for plot in plots:
            # Error checks
            if 'plot_type' not in plot:
                raise ValueError("Must specify 'plot_type' parameter.")
            plot_type = plot['plot_type']
            kwargs = plot.copy()
            del kwargs['plot_type']
            ax = self.plot_results(plot_type, ax=ax, **kwargs)
            
        return ax
        
        
    def plot_array(self, plots_arr, axes=None):
        """Plot an array of subplots.
        
        Parameters:
            plots_arr (ndarray(list)): 2d array of lists of dictionaries 
                    respresenting plots in a certain configuration.
            axes (ndarray): Array of subplot axes to plot on. If None, create 
                    a subplots array with the same shape as plots_arr
        """
        raise DeprecationWarning("The plot_array function is buggy due to lots"
                                 "of nested lists. Please use the plot_list"
                                 "function instead.")
        
        m, n = np.shape(plots_arr)[:2]
        
        # Initialize figure
        if axes is not None:
            axis = np.ravel(axes)[0]
            plt.sca(axis)
            fig = plt.gcf()
        else:
            fig, axes = plt.subplots(m, n, figsize=(10*n,10*m))
            
        # Reshape the respective arrays for easier iteration
        plots = plots_arr.reshape((1, -1))[0]
        axes_list = np.ravel(axes)
        
        for i, ax in enumerate(axes_list):
            self.plot_list(plots[i], ax=ax)
            
        return fig, axes
        
        
    def optimize(self, num_points, sample, delayed, points=None, 
                 print_vals='loss', clear_data=False, **kwargs):
        """Quick start function for optimization. Initializes points, computes 
        save values, and prints loss/gradient values.
        
        Parameters:
            num_points(int): number of points to initialize
            sample(str): 'random', 'same', 'grid', or 'given'
            delayed: whether to calculate for delayed, undelayed, or 'both'
            points(list): list of points to use for sample='given'
            print_vals(bool): whether to print gradient and loss information
            clear_data(bool): whether to clear the data at the end
        """
        # Initialize variables, initialize points, and run optimization
        self.initialize_vars(**kwargs)
        self.initialize_points(num_points, sample, points)
        self.calculate_save_values(delayed, **kwargs)
        
        self.print_vals(print_vals, delayed)       # Print specified values
        
        if clear_data is True:                     # Clear data
            self.clear()
            
            
    def get_mean_loss(self, delayed):
        if delayed is True:
            return np.mean(self.del_final_losses)
        else:
            return np.mean(self.final_losses)
        
        
    def print_vals(self, vals, delayed):
        if vals is True:
            self.print_loss(delayed)
            self.print_grad(delayed)
        elif vals == 'loss':
            self.print_loss(delayed)
        elif vals == 'grad':
            self.print_grad(delayed)
    
            
    def print_loss(self, delayed):
        if delayed == 'both':
            self.print_loss(False)
            self.print_loss(True)
        elif delayed is True:
            print("Minimum Delayed Loss:", np.min(self.del_final_losses))
            print("Mean Delayed Loss:", np.mean(self.del_final_losses))
            print("Median Delayed Loss:", np.median(self.del_final_losses))
        elif delayed is False:
            print("Minimum Undelayed Loss:", np.min(self.final_losses))
            print("Mean Undelayed Loss:", np.mean(self.final_losses))
            print("Median Undelayed Loss:", np.median(self.final_losses))
            
        
    def print_grad(self, delayed):
        if delayed == 'both':
            self.print_grad(False)
            self.print_grad(True)
        elif delayed is True:
            print("Minimum Delayed Gradient:", np.min(self.del_final_grads))
            print("Mean Delayed Gradient:", np.mean(self.del_final_grads))
            print("Median Delayed Gradient:", np.median(self.del_final_grads))
        elif delayed is False:
            print("Minimum Undelayed Gradient:", np.min(self.final_grads))
            print("Mean Undelayed Gradient:", np.mean(self.final_grads))
            print("Median Undelayed Gradient:", np.median(self.final_grads))
        
     
    def save_vals(self, filename):
        """Save Analyzer attributes as a pickle file"""
        attr_dict = self.__dict__.copy()
        
        # Drop attributes from dictionary
        drops = ['loss', 'grad', 'final_states', 'final_losses', 'final_grads',
                 'del_final_states', 'del_final_losses', 'del_final_grads']
        for attr in drops:
            attr_dict.pop(attr)
        
        # Save
        with open(filename, 'wb') as file:
            pickle.dump(attr_dict, file)
            
    
    def load_vals(self, filename):
        """Load Analyzer attributes from a pickle file"""
        # Load attribute dictionary
        with open(filename, 'rb') as file:
            attr_dict = pickle.load(file)
            
        # Check that file function data matches Analyzer instance
        if (attr_dict['loss_name'] == self.loss_name and attr_dict['n'] 
            == self.n):
            self.__dict__.update(attr_dict)
        else:
            raise ValueError("Functions from Analyzer instance and file data "
                             "do not match. Data was not loaded.")
        
        # Interpolate final values
        if len(self.time_series) != 0:
            self.__dict__['final_states'] = [self.time_series[i][-1] for i in 
                                             range(len(self.time_series))]
        if len(self.del_time_series) != 0:
            self.__dict__['del_final_states'] = [self.time_series[i][-1] for i 
                                                 in range(len(
                                                     self.del_time_series))]
        if len(self.loss_vals) != 0:
            self.__dict__['final_losses'] = [self.loss_vals[i][-1] for i in 
                                             range(len(self.loss_vals))]
        if len(self.del_loss_vals) != 0:
            self.__dict__['del_final_losses'] = [self.loss_vals[i][-1] for i in 
                                                 range(len(
                                                     self.del_loss_vals))]
        if len(self.loss_vals) != 0:
            self.__dict__['final_grads'] = [self.loss_vals[i][-1] for i in 
                                             range(len(self.grad_vals))]
        if len(self.del_loss_vals) != 0:
            self.__dict__['del_final_grads'] = [self.loss_vals[i][-1] for i in 
                                                 range(len(
                                                     self.del_grad_vals))]
            
            
    def compress(self, filename):
        """Save Analyzer attributes to a compressed .pbz2 file"""
        attr_dict = self.__dict__.copy()
        
        # Drop attributes from dictionary
        drops = ['loss', 'grad', 'final_states', 'final_losses', 'final_grads',
                 'del_final_states', 'del_final_losses', 'del_final_grads']
        for attr in drops:
            attr_dict.pop(attr)
        
        # Compress and save
        with bz2.BZ2File(filename, 'w') as file: 
            cPickle.dump(attr_dict, file)
            
            
    def decompress(self, filename):
        """Extract and load data from a compressed .pbz2 file"""
        # Load attribute dictionary
        data = bz2.BZ2File(filename, 'rb')
        attr_dict = cPickle.load(data)
            
        # Check that file function data matches Analyzer instance
        if (attr_dict['loss_name'] == self.loss_name and attr_dict['n'] 
            == self.n):
            self.__dict__.update(attr_dict)
        else:
            raise ValueError("Analyzer instance and file data functions do not"
                             " match. Data was not loaded.")
        
        # Interpolate final values
        if len(self.time_series) != 0:
            self.__dict__['final_states'] = [self.time_series[i][-1] for i in 
                                             range(len(self.time_series))]
        if len(self.del_time_series) != 0:
            self.__dict__['del_final_states'] = [self.time_series[i][-1] for i 
                                                 in range(len(
                                                     self.del_time_series))]
        if len(self.loss_vals) != 0:
            self.__dict__['final_losses'] = [self.loss_vals[i][-1] for i in 
                                             range(len(self.loss_vals))]
        if len(self.del_loss_vals) != 0:
            self.__dict__['del_final_losses'] = [self.loss_vals[i][-1] for i in 
                                                 range(len(
                                                     self.del_loss_vals))]
        if len(self.loss_vals) != 0:
            self.__dict__['final_grads'] = [self.loss_vals[i][-1] for i in 
                                             range(len(self.grad_vals))]
        if len(self.del_loss_vals) != 0:
            self.__dict__['del_final_grads'] = [self.loss_vals[i][-1] for i in 
                                                 range(len(
                                                     self.del_grad_vals))]
        
        
    def delete_initials(self):
        """Deletes all initialized points"""
        self.x_inits = None
        self.grid = None
            
        
    def delete_time_series(self):
        """Deletes all calculated undelayed time series"""
        self.time_series = list()
        self.final_states = list()
        
        
    def delete_loss(self):
        """Deletes all computed loss lists for undelayed optimization"""
        self.loss_vals = list()
        self.final_losses = list()
        
        
    def delete_grad(self):
        """Deletes all computed gradient lists for undelayed optimization"""
        self.grad_vals = list()
        self.final_grads = list()
        
        
    def delete_iters(self):
        """Resets iteration values for undelayed optimization"""
        self.iters = list()
    
    
    def delete_del_time_series(self):
        """Deletes all calculated delayed time series"""
        self.del_time_series = list()
        self.del_final_states = list()
        
        
    def delete_del_loss(self):
        """Deletes all computed loss lists for delayed optimization"""
        self.del_loss_vals = list()
        self.del_final_losses = list()
        
        
    def delete_del_grad(self):
        """Deletes all computed gradient lists for delayed optimization"""
        self.del_grad_vals = list()
        
        
    def delete_del_iters(self):
        """Resets iteration values for delayed optimization"""
        self.del_iters = list()    
        
        
    def delete_undelayed_data(self):
        """Deletes all computed undelayed data"""
        self.delete_time_series()
        self.delete_loss()
        self.delete_grad()
        self.delete_iters()
        self.conv = list()
        
        
    def delete_delayed_data(self):
        """Deletes all computed delayed data"""
        self.delete_del_time_series()
        self.delete_del_loss()
        self.delete_del_grad()
        self.delete_del_iters()
        self.del_conv = list()
    
        
    def clear(self):
        """Resets the helper by deleting the initial points and computed 
        values
        """
        self.delete_initials()
        self.delete_undelayed_data()
        self.delete_delayed_data()
        
        
    def __del__(self):
        self.clear()
        
