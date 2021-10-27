# Analyzer.py 

import numpy as np
import itertools
from matplotlib import pyplot as plt
from matplotlib import ticker
from Optimizer_Scripts.learning_rate_generator import generate_learning_rates
from Optimizer_Scripts import Delayer
from Optimizer_Scripts import optimizers 
from Optimizer_Scripts import functions

class Analyzer:
    """Class to analyze the effects of delayed optimization on various functions.
    Includes functions to calculate the time series, loss values, and gradient 
    values over time, and to graph plots of that data
    """
    def __init__(self, n, loss_name, optimizer_name='Adam', max_L=1, num_delays=1000, 
                 maxiter=2000, tol=1e-5, compute_loss=True, save_grad=True):
        """The initializer for the Helper class.
            
           Parameters: 
               n(int): the dimension of the state vector
               loss_name(str): name of the loss function to be analyzed
               optimizer_name(str): name of the optimization algorithm to be used
               max_L(int): the maximum delay of the system
               num_delays(int): the number of delays to compute with the system
               maxiter(int): the maximum number of iterations in the system
               tol(float): the tolerance value used in the system
               compute_loss(bool): whether the Delayer class should compute the loss values
               save_grad(bool): whether the Delayer class should save gradient values
        """
        # Attributes for the Analyzer class
        self.n = n
        self.loss_name = loss_name
        self.optimizer_name = optimizer_name
        self.initialize_functions()
        self.initialize_params()
        self.max_L = max_L
        self.num_delays = num_delays
        self.maxiter = maxiter
        self.tol = tol
        self.compute_loss = compute_loss
        self.save_grad = save_grad
        
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
        """Initializes the loss and gradient functions based on loss_name. Also initilizes 
        the default range grid and the known minimizer for the function."""
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
            self.grad = functions.rastrigin_deriv_gen(self.n)
            self.range_grid = [-32.,32.]
            self.minimizer = np.zeros(self.n)
        else:
            raise ValueError("The '{}' function has not been implemented.".format(self.loss_name))
        
        
    def initialize_params(self):
        """Set the delayed and undelayed optimal parameters closest to those computed.
        """
        if self.loss_name == 'Rosenbrock':
            if self.n == 2:
                self.params = {'step_size': 1800, 'min_learning_rate': 0.16862560897303366, 'max_learning_rate':  3.3025870984923507}
                self.del_params = {'step_size': 100, 'min_learning_rate': 0.5427218194366088, 'max_learning_rate': 2.917445299778115}
            elif self.n == 10:
                self.params = {'step_size': 1800, 'min_learning_rate': 0.8007229432536243, 'max_learning_rate': 2.2449481296632032}
                self.del_params = {'step_size': 1600, 'min_learning_rate': 0.1063467709571041, 'max_learning_rate': 2.013029349056361}
            elif self.n == 100:
                self.params = {'step_size': 1700, 'min_learning_rate': 0.517023608040682, 'max_learning_rate': 2.663301806828187}
                self.del_params = {'step_size': 1700, 'min_learning_rate': 0.6663565546501306, 'max_learning_rate': 1.542771215892512}
            elif self.n == 1000:
                self.params = {'step_size': 1900, 'min_learning_rate': 0.7913344486193052, 'max_learning_rate': 3.153253937908706}
                self.del_params = {'step_size': 2200, 'min_learning_rate': 0.3938785608353136, 'max_learning_rate': 1.9686044621631718}
            elif self.n == 10000:
                self.params = {'step_size': 200, 'min_learning_rate': 0.051169248252087185, 'max_learning_rate': 2.52094246796635}
                self.del_params = {'step_size': 2000, 'min_learning_rate': 0.3531031777251941, 'max_learning_rate': 2.8043327181669144}
            else:
                self.params = {'step_size': 2000, 'min_learning_rate': 0.5, 'max_learning_rate': 3.0}
                self.del_params = {'step_size': 1800, 'min_learning_rate': 0.5, 'max_learning_rate': 2.0}
        elif self.loss_name == 'Zakharov':
            if self.n == 2:
                self.params = {'step_size': 1600, 'min_learning_rate': 0.820160859068515, 'max_learning_rate': 2.621539273761733}
                self.del_params = {'step_size': 600, 'min_learning_rate': 0.9002716481735573, 'max_learning_rate': 3.9453205411254526}
            elif self.n == 10:
                self.params = {'step_size': 1300, 'min_learning_rate': 0.8755669260416792, 'max_learning_rate': 2.0703992165503715}
                self.del_params = {'step_size': 1100, 'min_learning_rate': 0.5981517265695458, 'max_learning_rate': 1.7721689810132744}
            elif self.n == 100:
                self.params = {'step_size': 500, 'min_learning_rate': 0.08078391698353116, 'max_learning_rate': 3.049427241295275}
                self.del_params = {'step_size': 900, 'min_learning_rate': 0.5920671826485262, 'max_learning_rate': 3.7103587561074676}
            elif self.n == 1000:
                self.params = {'step_size': 800, 'min_learning_rate': 0.2328851755973872, 'max_learning_rate': 3.3296254127502167}
                self.del_params = {'step_size': 900, 'min_learning_rate': 0.5882282586803385, 'max_learning_rate': 3.5049523306448735}
            elif self.n == 10000:
                self.params = {'step_size': 200, 'min_learning_rate': 0.7746451422263353, 'max_learning_rate': 2.580862669000447}
                self.del_params = {'step_size': 400, 'min_learning_rate': 0.46155695597287816, 'max_learning_rate': 1.5510060856536165}
            else:
                self.params = {'step_size': 500, 'min_learning_rate': 0.7, 'max_learning_rate': 2.6}
                self.del_params = {'step_size': 1000, 'min_learning_rate': 0.5, 'max_learning_rate': 3.0}
        else:
            self.params = {'step_size': 740, 'min_learning_rate': 0.23, 'max_learning_rate': 2.98}
            self.del_params = self.params
        
        
    def initialize_optimizer(self, delayed, beta_1=0.9, beta_2=0.999):
        """Initializes the optimial paramters and the optimizer."""
        params = dict()
        params['beta_1'] = beta_1
        params['beta_2'] = beta_2
    
        if delayed is True:
            params['learning_rate'] = generate_learning_rates(False, self.del_params)
        else:
            params['learning_rate'] = generate_learning_rates(False, self.params)
        if self.optimizer_name == 'Adam':
            self.optimizer = optimizers.Adam(params)
        else:
            raise ValueError("The '{}' optimizer has not been implemented.".format(self.optimizer_name))
            
            
    def initialize_delayer(self):
        """Initialize the Delayer class item for optimization."""
        self.delayer = Delayer.Delayer(self.n, self.optimizer, self.loss, self.grad, max_L=self.max_L, 
                                       num_delays=self.num_delays, compute_loss=self.compute_loss, 
                                       save_grad=self.save_grad)
            
            
    def initialize_points(self, num_points, sample, points=None, create_mesh=True):
        """Initialize the initial points for the optimization.
        
        Parameters:
            num_points(int): number of points to initialize
            sample(str): 'random' to choose random initial points
                         'same' to have all be the same randomly chosen point 
                         'grid' to create an evenly space grid of num_points^2 points
                         'given' to use the given points
            points(list): list of points to use for sample='given'
        """
        # Reset previous points, delete saved data
        self.clear()
        
        # Set the points on which to optimize
        if sample == 'random':
            self.x_inits = np.random.uniform(self.range_grid[0], self.range_grid[1], size=(num_points,self.n))
        elif sample == 'grid':
            self.x_inits, self.grid = self.create_grid(num_points, create_mesh)
        elif sample == 'same':
            xint = np.random.uniform(self.range_grid[0], self.range_grid[1], size=self.n)
            self.x_inits = np.tile(xint, (num_points, 1))
        elif sample == 'given':
            if points is None:
                raise ValueError("Must provide points for type_test='given'.")
            self.x_inits = np.asarray(points)
        else:
            raise ValueError("Test type '{}' does not exist.".format(sample))
    
    
    def create_grid(self, num_points, create_mesh=True):
        """Helper function used to initialize an evenly spaced grid of initial points.
        
        Parameters:
            num_points(int): number of points to use for each dimension of the grid
                             (total number of points is num_points^n)
        Returns:
            x_inits(ndarray): an array representing the initial points of the grid
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
        
        if create_mesh is True:
            X, Y = np.meshgrid(x, x)
            grid = [X, Y]
        
        return x_inits, grid
    
            
    def initialize_vars(self, range_grid=None, max_L=None, num_delays=None, maxiter=None, tol=None):
        """Updates parameters if new values are given."""
        if range_grid is not None:
            self.range_grid = range_grid
        if max_L is not None:
            self.max_L = max_L
        if num_delays is not None:
            self.num_delays = num_delays
        if maxiter is not None:
            self.maxiter = maxiter
        if tol is not None:
            self.tol = tol
            
            
    def run_single_start(self, x_init, use_delays, D=None, random=True, break_opt=True):
        """Runs the optimization for a single point."""
        self.delayer.x_init = x_init
        self.delayer.compute_time_series(tol=self.tol, maxiter=self.maxiter, use_delays=use_delays, 
                                         random=random, D=D, break_opt=break_opt)
            
            
    def calculate_save_values(self, delayed, max_L=None, num_delays=None, maxiter=None, tol=None, 
                              D=None, random=True, break_opt=True, save_state=True, save_loss=True, 
                              save_grad=True, save_iters=True):
        """Run the optimization on the initial points already initialized and saves 
        values to be plotted.
        
        Parameters:
            delayed: whether to delay the optimization or not ('both' calculates both)
            max_L(int): the maximum delay of the system
            num_delays(int): the number of delays to compute with the system
            maxiter(int): the maximum number of iterations in the system
            tol(float): the tolerance value used in the system
            D(list): list of deay distributions for the Delayer class
            random(bool): whether delays should be stochastic or not
            break_opt(bool): whether optimization should stop when convergence criteria is met
            save_state(bool): whether the class should save the time series of each point
            save_loss(bool): whether the class should save loss values over time for each point
            save_grad(bool): whether the class should save gradient values over time for each point
            save_iters(bool): whether the class should save the number of iterations each point takes 
                              to converge
        """
        if delayed == 'both':
            # If 'both' run the function for both True and False
            self.calculate_save_values(False, max_L, num_delays, maxiter, tol, D, random, 
                                       break_opt, save_state, save_loss, save_grad, save_iters)
            self.calculate_save_values(True, max_L, num_delays, maxiter, tol, D, random, 
                                       break_opt, save_state, save_loss, save_grad, save_iters)
        elif delayed in (True, False):
            # Reset old data
            if delayed is True:
                self.delete_delayed_data()
            else:
                self.delete_undelayed_data()
                
            # Initialize
            self.initialize_vars(max_L=max_L, num_delays=num_delays, maxiter=maxiter, tol=tol)
            self.initialize_optimizer(delayed)
            self.initialize_delayer()
            
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
                    
                # Recreate the delayer and optimizer to reset the learning rate generator
                del self.delayer
                del self.optimizer
                self.initialize_optimizer(delayed)
                self.initialize_delayer()
            del self.delayer
            del self.optimizer
            
        else:   # Check for incorrect inputs
            raise ValueError("Variable 'delays' must be 'both', True, or False.")
        return
            
            
    def extract_values(self, delayed, focus):
        """Uses the delayed boolean and focus string to return the desired data.
        
        Parameters:
            delayed(bool): whether the desired data is for delayed or undelayed
            focus(str): 'loss', 'grad', 'state', or 'iters' values to return
            
        Returns:
            (list(list)): requested values over time for all points
            (list): final values for each point
            (str): string used in creating plot titles
        """
        if delayed is True:
            if focus == 'state':
                return self.del_time_series, self.del_final_states, "Delayed State"
            if focus == 'loss':
                return self.del_loss_vals, self.del_final_losses, "Delayed Loss"
            if focus == 'grad':
                return self.del_grad_vals, self.del_final_grads, "Delayed Gradient"
            if focus == 'iters':
                return None, self.del_iters, "Delayed Iteration"
        if delayed is False:
            if focus == 'state':
                return self.time_series, self.final_states, "State"
            if focus == 'loss':
                return self.loss_vals, self.final_losses, "Loss"
            if focus == 'grad':
                return self.grad_vals, self.final_grads, "Gradient"
            if focus == 'iters':
                return None, self.iters, "Iteration"
            

    def set_bins(self, num_bins, values, del_values):
        """Sets the bin size equally for delayed and undelayed final values"""
        min_val = min(min(values), min(del_values))
        max_val = max(max(values), max(del_values))
        return np.linspace(min_val, max_val, num_bins)
        
            
    def extract_dims(self, point_ind, dim_tuple, delayed):
        """Pulls the desired data from the saved values."""
        data = list()
        if delayed is True:
            data.append(self.del_time_series[point_ind][1:,dim_tuple[0]].tolist())
            data.append(self.del_time_series[point_ind][1:,dim_tuple[1]].tolist())
        else: 
            data.append(self.time_series[point_ind][1:,dim_tuple[0]].tolist())
            data.append(self.time_series[point_ind][1:,dim_tuple[1]].tolist())
        return data
        
    
    def ravel_data(self, values):
        """Extracts the nested list of values into a single list."""
        new_list = list()
        for data in values:
            for element in data:
                new_list.append(element)
        return new_list
    
    
    def set_bounds(self, values, time_plot):
        """Computes and returns the vmax and vmin values for plotting."""
        vmax, vmin = np.nanmax(np.nanmax(values)), np.nanmin(np.nanmin(values))
        if vmin > 0 or time_plot is True:
            vmin = 0
        return vmax, vmin
    
    
    def plot_colorbar(self, fig, axis, image):
        colorbar = fig.colorbar(image, ax=axis)
        colorbar.set_alpha(1)
        colorbar.draw_all()
        
            
    def plot_results(self, delayed, type_plot, focus, num_bins=25, fixed_bins=True, 
                     plot_dims=[(0,1)], time_plot=False, colorbar=True, fixed_limits=True, 
                     contour_plot=False, include_exteriors=False, cmap='winter', 
                     cmap2='autumn', title=None):
        """Plot the previously computed results.
         
           Parameters: 
               delayed - whether to plot delayed values (True), undelayed values (False), or 'both'
               type_plot(str) - 'finals' for final state, loss, gradient, or iteration values
                                'path' to plot the paths of points 
                                'basin' for basin of attraction plots
               focus(str) - value to be plotted ('state', 'loss', 'grad', 'iters')
               num_bins(int) - the number of bins used [finals]
               fixed_bins(bool) - should the bins be the same for the delayed and undelayed 
                                  histograms [finals]
               plot_dims(list(tuples)) - the list of dimensions to plot against each other [path]
               time_plot(bool) - whether to plot the time series of each point [path]
               fixed_limits(bool) - whether to fix the limits of the graph to the range_grid or
                                    let the program pick its own limits [path]
               contour_plot(bool) - whether to plot the contour of the function on top of the plot
                                    [basin]
               include_exteriors(bool) - whether to plot the values for points that did not converge
                                         [basin, iters only]
        """
        # Error checker
        if delayed not in ('both', True, False):
            raise ValueError("Variable 'delays' must be 'both', True, or False.")
        if type_plot not in ('finals', 'path', 'basin'):
            raise ValueError("Plot type '{}' does not exist.".format(type_plot))
        if focus not in ('state', 'loss', 'grad', 'iters'):
            raise ValueError("Plot focus must be 'state', 'loss', 'grad', or 'iters', not {}.".format(focus))
        
        if type_plot == 'finals':
            # Initialize for the histogram
            fig, ax = plt.subplots(1, 1, figsize=(10,8))
                                   
            if delayed == 'both':    
                alpha = 0.3
                final_vals, type_str = self.extract_values(False, focus)[1:]
                del_final_vals = self.extract_values(True, focus)[1]
                if focus == 'state':
                    final_vals = self.ravel_data(final_vals)
                    del_final_vals = self.ravel_data(del_final_vals)
                if fixed_bins is True:
                    bins = self.set_bins(num_bins, final_vals, del_final_vals)
                else:
                    bins = num_bins
                
                ax.hist(final_vals, bins=bins, alpha=alpha, color='b')
                ax.hist(del_final_vals, bins=bins, alpha=alpha, color='r')
                    
            else:
                alpha = 0.8
                if delayed is True:
                    color = 'r'
                else:
                    color = 'b'
                final_vals, type_str = self.extract_values(delayed, focus)[1:]
                if focus == 'state':
                    final_vals = self.ravel_data(final_vals)
                ax.hist(final_vals, bins=num_bins, alpha=alpha, color=color)
                
            if title is None:
                title = "Final {} Values".format(type_str)
                
        elif type_plot == 'path':
            # Check that the focus is correct
            if focus in ('state', 'iters'):
                raise ValueError("Path plot type must have a focus of 'loss' or 'grad'.")
                
            # Initialize for the graph
            num_plots = len(plot_dims)
            fig, ax = plt.subplots(num_plots, 1, figsize=(10,8*(num_plots)))
            if type(ax) is not np.ndarray:
                ax = np.array([ax])
            alpha = 0.01
            
            if delayed == 'both':
                values, final_values, type_str = self.extract_values(False, focus)
                del_values, del_final_values = self.extract_values(True, focus)[:2]
                vmax, vmin = self.set_bounds(values, time_plot)
                del_vmax, del_vmin = self.set_bounds(del_values, time_plot)
                
                for j in range(num_plots):
                    axis = ax[j]
                    dim_tuple = plot_dims[j]
                    for i in range(len(self.x_inits)):
                        data = self.extract_dims(i, dim_tuple, False)
                        del_data = self.extract_dims(i, dim_tuple, True)
                        
                        im = axis.scatter(data[0], data[1], c=values[i], alpha=alpha, cmap=cmap, s=70, vmax=vmax, vmin=vmin)
                        axis.scatter(del_data[0], del_data[1], c=del_values[i], alpha=alpha, cmap=cmap2, s=70, vmax=del_vmax, vmin=del_vmin)
                        if time_plot is True:
                            axis.plot(data[0], data[1], color='b', alpha=0.3)
                            axis.plot(del_data[0], del_data[1], color='r', alpha=0.3)
                    if colorbar is True:
                        self.plot_colorbar(fig, axis, im)
                    axis.set_xlabel("Dimension {}".format(dim_tuple[0]))
                    axis.set_ylabel("Dimension {}".format(dim_tuple[1]))
                    if fixed_limits is True:
                        axis.set_xlim(self.range_grid)
                        axis.set_ylim(self.range_grid)
            else:
                values, final_values, type_str = self.extract_values(delayed, focus)
                vmax, vmin = self.set_bounds(values, time_plot)
                color = 'b'
                if delayed is True:
                    cmap = cmap2
                    color = 'r'
                    
                for j in range(num_plots):
                    axis = ax[j]
                    dim_tuple = plot_dims[j]
                    for i in range(len(self.x_inits)):
                        data = self.extract_dims(i, dim_tuple, delayed)
                        im = axis.scatter(data[0], data[1], c=values[i], alpha=alpha, cmap=cmap, s=70, vmax=vmax, vmin=vmin)
                        if time_plot is True:
                            axis.plot(data[0], data[1], color=color, alpha=0.3)
                    if colorbar is True:
                        self.plot_colorbar(fig, axis, im)
                    axis.set_xlabel("Dimension {}".format(dim_tuple[0]))
                    axis.set_ylabel("Dimension {}".format(dim_tuple[1]))
                    if fixed_limits is True:
                        axis.set_xlim(self.range_grid)
                        axis.set_ylim(self.range_grid)
                        
            
            if title is None:
                title = "Path Tracker on the {} function of {} dimensions".format(self.loss_name, self.n)
                    
        elif type_plot == 'basin':
            # Check that the focus is correct
            if focus == 'state':
                raise ValueError("Basin plot type must have a focus of 'loss', 'grad', or 'iters'.")

            # Check that points were computed as a grid
            if self.grid is None:
                raise ValueError("Basin plot type is only compatible with 'grid' test type.")
            
            # Check that function is 2D
            if self.n != 2:
                raise NotImplementedError("Basin plot type is only implemented for functions of dimension 2.")
            
            if delayed == 'both':
                fig, ax = plt.subplots(1, 2, figsize=(20,8))
                final_values, type_str = self.extract_values(False, focus)[1:]
                del_final_values = self.extract_values(True, focus)[1]
                num_points = len(self.x_inits)
                X, Y = self.grid
                
                Z, del_Z = np.zeros(num_points), np.zeros(num_points)
                for i in range(num_points):
                    Z[i], del_Z[i] = final_values[i], del_final_values[i]
                    if include_exteriors is False and focus == 'iters':    # Don't include points that didn't converge
                        if self.conv[i] is False:  
                            Z[i] = np.nan
                        if self.del_conv[i] is False:
                            del_Z[i] = np.nan
                
                Z = np.resize(Z, (len(X),len(Y))).T
                del_Z = np.resize(del_Z, (len(X),len(Y))).T
                ax[0].patch.set_color('.25')
                vmax, vmin = self.set_bounds(np.stack([Z,del_Z]), False)
                im0 = ax[0].contourf(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax)
                ax[0].set_xlabel("Dimension 0")
                ax[0].set_ylabel("Dimension 1")
                ax[0].set_title("Undelayed")
                ax[1].patch.set_color('.25')
                im1 = ax[1].contourf(X, Y, del_Z, cmap=cmap, vmin=vmin, vmax=vmax)
                ax[1].set_xlabel("Dimension 0")
                ax[1].set_ylabel("Dimension 1")
                ax[1].set_title("Delayed")
                
                if colorbar is True:
                    self.plot_colorbar(fig, ax[0], im0)
                    self.plot_colorbar(fig, ax[1], im1)
                
                if contour_plot is True:
                    if num_points < 250:
                        num_points = 250
                    points, grid = self.create_grid(num_points)
                    X, Y = grid
                    Z = np.zeros(num_points**2)
                    
                    for i in range(num_points**2):
                        Z[i] = self.loss(points[i])
                    Z = np.resize(Z, (len(X),len(Y))).T
                    ax[0].contour(X, Y, Z, locator=ticker.LogLocator(), cmap=cmap2)
                    ax[1].contour(X, Y, Z, locator=ticker.LogLocator(), cmap=cmap2)
                
            else:
                # Initialize values and the figure
                num_plots = len(plot_dims)
                num_points = len(self.x_inits)
                X, Y = self.grid
                fig, ax = plt.subplots(num_plots, 1, figsize=(10,8*(num_plots)))
                if type(ax) is not np.ndarray:
                    ax = np.array([ax])    # TODO : Make sure that casting as an array doesnt create problems
                    
                if focus in ['loss','grad']:    # Always include exteriors for loss and gradient
                    include_exteriors = True
                
                
                ## Need to find the indices of points corresponding to each set of dimensions we want to graph
                final_values, type_str = self.extract_values(delayed, focus)[1:]
                
                
                #print(num_points)
                #for j in range(num_plots):
                #    axis = ax[j]
                #    dim_tuple = plot_dims[j]
                    
                #    Z = np.zeros(num_points)
                    
                    #im = axis.contourf(X, Y, Z, cmap=cmap)
                        
                        
                #    if colorbar is True:
                #        self.plot_colorbar(fig, axis, im)
                #    axis.set_xlabel("Dimension {}".format(dim_tuple[0]))
                #    axis.set_ylabel("Dimension {}".format(dim_tuple[1]))
                #    if fixed_limits is True:
                #        axis.set_xlim(self.range_grid)
                #        axis.set_ylim(self.range_grid)
                
                
                
                
                for k, axis in enumerate(ax):
                    Z = np.zeros(num_points)
                    for i in range(num_points):
                        Z[i] = final_values[i]
                        if include_exteriors is False:    # Do not include points that did not converge
                            if delayed is True:
                                if self.del_conv[i] is False:  
                                    Z[i] = np.nan
                            else:
                                if self.conv[i] is False:
                                    Z[i] = np.nan
                    
                    Z = np.resize(Z, (len(X),len(Y))).T
                    axis.patch.set_color('.25')
                    im = axis.contourf(X, Y, Z, cmap=cmap)
                    axis.set_xlabel("Dimension {}".format(plot_dims[k][0]))
                    axis.set_ylabel("Dimension {}".format(plot_dims[k][1]))
                    
                    if colorbar is True:
                        self.plot_colorbar(fig, axis, im)
                    
                if contour_plot is True:
                    if num_points < 250:
                        num_points = 250
                    points, grid = self.create_grid(num_points)
                    X, Y = grid
                    Z = np.zeros(num_points**2)
                    
                    for i in range(num_points**2):
                        Z[i] = self.loss(points[i])
                    Z = np.resize(Z, (len(X),len(Y))).T
                    ax.contour(X, Y, Z, locator=ticker.LogLocator(), cmap=cmap2)
            
            if title is None:
                title = "{} Basin of Attraction Plot".format(type_str)
            
        fig.suptitle(title, size='xx-large')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        plt.show()
        
            
    def plot_list(self, plots, num_bins=25, fixed_bins=True, plot_dims=[(0,1)], time_plot=False,
                  colorbar=True, fixed_limits=True, contour_plot=False, include_exteriors=False, 
                  cmap='winter', cmap2='autumn'):
        """Unpacks a list of tuples, (delayed, type_plot, focus), representing plots with 
        those parameters and plots each one
        """
        for plot in plots:
            delayed, type_plot, focus = plot
            self.plot_results(delayed, type_plot, focus, num_bins, fixed_bins, plot_dims, 
                              time_plot, colorbar, fixed_limits, contour_plot, include_exteriors, 
                              cmap, cmap2)
        
        
    def optimize(self, num_points, sample, delayed, plots=[], points=None, create_mesh=True, 
                 range_grid=None, max_L=None, num_delays=None, maxiter=None, tol=None, D=None, 
                 random=True, break_opt=True, save_state=True, save_loss=True, save_grad=True, 
                 save_iters=True, num_bins=25, fixed_bins=True, plot_dims=[(0,1)], time_plot=False, 
                 colorbar=True, fixed_limits=True, contour_plot=False, include_exteriors=False, 
                 cmap='winter', cmap2='autumn', print_loss=True, print_grad=False, clear_data=True):
        """Contains all the basic functions of the other major functions. Initializes
        points, computes save values, and plots the data according to the parameters.
        
        Parameters:
            num_points(int): number of points to initialize
            sample(str): 'random', 'same', 'grid', or 'given'
            delayed: whether to calculate for delayed, undelayed, or 'both'
            plots(list): list of tuples of plots (delayed, type_plot, focus)
            points(list): list of points to use for sample='given'
            print_vals(bool): whether to print gradient and loss information
            clear_data(bool): whether to clear the data at the end
        """
        self.initialize_vars(range_grid=range_grid) # Other values are reinitialized in calculate_save_values
        self.initialize_points(num_points, sample, points, create_mesh)
        self.calculate_save_values(delayed, max_L, num_delays, maxiter, tol, D, random, 
                                   break_opt, save_state, save_loss, save_grad, save_iters)
        self.plot_list(plots, num_bins, fixed_bins, plot_dims, time_plot, colorbar, 
                       fixed_limits, contour_plot, include_exteriors, cmap, cmap2)
        
        if print_loss is True:
            self.print_loss(delayed)
        if print_grad is True:
            self.print_grad(delayed)
        
        if clear_data is True:
            self.clear()
        
            
    def print_loss(self, delayed):
        if delayed == 'both':
            self.print_loss(False)
            self.print_loss(True)
        elif delayed is True:
            print("Minimum Delayed Loss:", np.min(self.del_final_losses))
            print("Mean Delayed Loss:", np.mean(self.del_final_losses))
            print("Median Delayed Loss:", np.median(self.del_final_losses))
        elif delayed is False:
            print("Minimum Loss:", np.min(self.final_losses))
            print("Mean Loss:", np.mean(self.final_losses))
            print("Median Loss:", np.median(self.final_losses))
            
        
    def print_grad(self, delayed):
        if delayed == 'both':
            self.print_grad(False)
            self.print_grad(True)
        elif delayed is True:
            print("Minimum Delayed Gradient:", np.min(self.del_final_grads))
            print("Mean Delayed Gradient:", np.mean(self.del_final_grads))
            print("Median Delayed Gradient:", np.median(self.del_final_grads))
        elif delayed is False:
            print("Minimum Gradient:", np.min(self.final_grads))
            print("Mean Gradient:", np.mean(self.final_grads))
            print("Median Gradient:", np.median(self.final_grads))
        
        
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
        """Resets the helper by deleting the initial points and computed values"""
        self.delete_initials()
        self.delete_undelayed_data()
        self.delete_delayed_data()
        
    def __del__(self):
        self.clear()
        
