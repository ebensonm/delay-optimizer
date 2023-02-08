# Analyzer_w_result.py 

import numpy as np
import warnings
import Optimizer_Scripts.learning_rate_generator as lrgen
from Optimizer_Scripts.Delayer import Delayer
from Optimizer_Scripts.Data import Data
from Optimizer_Scripts.LossFunc import LossFunc
from Optimizer_Scripts import optimizers


class FuncOpt:
    """Class to execute delayed or undelayed optimization on a given function"""
    def __init__(self, loss_name, dim):
        """Initializer for the FuncOpt class
        
        Parameters: 
            loss_name(str): name of the loss function to be analyzed
            dim(int): the dimension of the state vector
        """
        self.loss_func = LossFunc(loss_name, dim)
                
                
    def random_points(self, num_points):
        """Randomly initialize given number of points within the domain"""
        points = np.random.uniform(*self.loss_func.domain, 
                                   size=(num_points,self.loss_func.n))
        self.x_inits = points
        
        
    def load_points(self, points):
        """Load initial optimization points from input"""
        points = np.asarray(points)
        if points.shape[1] != self.loss_func.n:
            raise ValueError("Points array dimension mismatch")
        self.x_inits = points
            
                
    def get_optimizer(self, optimizer_name, beta_1=0.9, beta_2=0.999, 
                      **lr_params):
        """Initialize parameters and return the optimizer object."""
        if optimizer_name == 'Adam':
            params = {
                'beta_1': beta_1, 
                'beta_2': beta_2,
                'learning_rate': lrgen.generate_learning_rates(**lr_params)
                }
            return optimizers.Adam(params)
        else:
            raise ValueError("Invalid optimizer name.")            
                
                
    def get_delayer(self, optimizer, delay_type, save_state=True, 
                    save_loss=True, save_grad=True):
        """Initialize and return the Delayer object for optimization."""
        if save_state: save_state = True  # Set True when save_state is a tuple
        return Delayer(self.loss_func.n, delay_type, self.loss_func.loss, 
                       self.loss_func.grad, optimizer, save_state, save_loss, 
                       save_grad)            
                
                
    def get_lr_params(self, lr_type, **kwargs):
        lr_keys = lrgen.get_param_dict(lr_type).keys()
        try:
            lr_params = {key : kwargs[key] for key in lr_keys}
            lr_params["lr_type"] = lr_type
            return lr_params
        except:
            raise KeyError(r"Learning rate type '{}' requires the following "
                           "keys: {}".format(lr_type, lr_keys))
        

    def optimize(self, delay_type, lr_type, optimizer_name='Adam', tol=1e-5, 
                 maxiter=5000, break_opt=True, save_state=(0,1), 
                 save_loss=True, save_grad=False, **lr_kwargs):
        """Run the optimization on the initial points already initialized and 
        saves values to be plotted.
        
        Parameters:
            delay_type(DelayType): class object containing delay parameters
            optimizer_name(str): the name of the optimizer to use
            maxiter(int): the maximum number of iterations for optimization
            tol(float): the tolerance value for optimization
            break_opt(bool): whether optimization should stop when convergence 
                             criteria is met
            save_state(bool/tuple): state dimensions to save during optimization
            save_loss(bool): whether to save loss values over time 
            save_grad(bool): whether to save gradient values over time
        """
        # Check if points have been initialized
        if len(self.x_inits) == 0:
            warnings.warn("No points have been initialized.")
            return
        
        # Initialize
        lr_params = self.get_lr_params(lr_type, **lr_kwargs)
        self.data = Data(self.loss_func)
        self.data.set_delay_scheme(delay_type, maxiter, tol, break_opt)
        self.data.set_optimizer_params(optimizer_name, lr_params)
            
        for x in self.x_inits:
            # Initialize optimizer and delayer for new point
            optimizer = self.get_optimizer(optimizer_name, **lr_params)
            delayer = self.get_delayer(optimizer, delay_type, save_state, 
                                       save_loss, save_grad)
            
            delayer.optimize(x, maxiter, tol, break_opt)
            self.data.add_point(delayer, save_state, save_loss, save_grad)
            
        self.data.close()   # Reformat data into arrays instead of lists
        return self.data


    def save_data(self, filename):
        self.data.save(filename)
        
        
    def load_data(self, filename):
        data = Data.load(filename)
        if (self.loss_func != data.get_loss_function()):
            raise ValueError("Functions do not match. Data was not loaded.")
        self.data = data
        
    
    @classmethod
    def load(cls, filename):
        """Load FuncOpt object from OptResult data"""
        data = Data.load(filename)
        obj = cls(data.loss_name, data.dim)
        obj.x_inits = data.get_x_inits()
        obj.data = data
        return obj
    
    
    def delete_initials(self):
        """Deletes all initialized points"""
        del self.x_inits
            
        
    def delete_data(self):
        """Deletes all optimization data but retains initial values"""
        del self.data
        
        
    def reset(self):
        """Resets all values (except loss function)"""
        self.delete_initials()
        self.delete_data()
        
    
        
