# Data.py

import numpy as np
from Optimizer_Scripts import DelayTypeGenerators
from Optimizer_Scripts.LossFunc import LossFunc
import pickle
import blosc


class Data:
    """Object to hold optimization data."""
    
    def __init__(self, loss_func):
        self.set_loss_function(loss_func)
        
        self.state_vals = []
        self.loss_vals = []
        self.grad_vals = []
        self.converged = []
        
    
    # Initialization -------------------------------------------------------
    
    def set_loss_function(self, loss_func):
        """Set values from loss function object"""
        self.loss_name = loss_func.loss_name
        self.dim = loss_func.n
        self.domain = loss_func.domain
        self.minimizer = loss_func.minimizer
        
    
    def set_delay_scheme(self, delay_type, maxiter, tol, break_opt):
        """Set delay scheme values from DelayType object"""
        self.delay_params = delay_type.__dict__
        self.maxiter = maxiter
        self.tol = tol
        self.break_opt = break_opt
        
        
    def set_optimizer_params(self, optimizer_name, lr_params):
        """Set parameter values for the optimizer"""
        self.optimizer_name = optimizer_name
        self.lr_params = lr_params
        
    
    # Optimization ---------------------------------------------------------
    
    def add_point(self, result, save_state, save_loss, save_grad):
        """Append the values for the delayed optimization of a single point"""
        if save_state is not False:
            self.state_vals.append(result.state_vals)
        if save_loss is True:
            self.loss_vals.append(result.loss_vals)
        if save_grad is True:
            self.grad_vals.append(result.grad_vals)
            
        self.converged.append(result.converged)
        
    
    # Data retrieval -------------------------------------------------------
    
    def get_loss_function(self):
        loss_func = LossFunc(self.loss_name, self.dim)
        loss_func.domain = self.domain
        loss_func.minimizer = self.minimizer
        return loss_func
    
    
    def get_delay_type(self):
        return DelayTypeGenerators.get_delay_type(self.delay_params)  
    
    
    def get_initials(self, value_list):
        """Returns an array of initial values from the given list of sequences"""
        return np.asarray([val[0] for val in value_list])
    
    
    def get_x_inits(self):
        return self.get_initials(self.state_vals)
    
    
    def get_finals(self, value_list):
        """Returns an array of final values from the given list of sequences"""
        return np.asarray([val[-1] for val in value_list])
    
    
    def get_mean_final(self, value_list):
        """Returns the final mean value of the given list of sequences"""
        return np.mean(self.get_finals(value_list), axis=0)
    
    
    def get_slice(self, dim_tuple):
        """Returns the desired slice of the state data. 
        
        Parameters:
            dim_tuple (tuple): The dimensions to extract
        Returns:
            (ndarray(list(ndarray))): Ragged nested array of sliced state 
                sequences
        """
        return np.array([np.array([it[np.r_[dim_tuple]] for it in point]) 
                         for point in self.state_vals], dtype=object)
    
    
    def get_loss_array(self):
        """Returns the full, nonragged 2d array of loss values. Array has 
        dimensions (num_points, maxiter+1) 
        """
        loss_arr = np.empty([len(self.loss_vals), self.maxiter+1])
        for j, loss_vals in enumerate(self.loss_vals):
            loss_arr[j][:len(loss_vals)] = loss_vals
            loss_arr[j][len(loss_vals):] = loss_vals[-1]
        
        return loss_arr    
        
    
    # Saving / Loading -----------------------------------------------------
    
    def close(self):
        """Reduce memory size of lists for storage"""
        self.state_vals = self.state_vals[:]
        self.loss_vals = self.loss_vals[:]
        self.grad_vals = self.grad_vals[:]
        self.converged = self.converged[:]
        
    
    def save(self, filename): 
        """Save data to given file"""
        self.close()
        
        if not filename.endswith('.dat'):   # Format filename
            filename += '.dat'
            
        pickled_data = pickle.dumps(self)
        compressed_pickle = blosc.compress(pickled_data)
        
        with open(filename, "wb") as file: 
            file.write(compressed_pickle)
        
        
    @classmethod
    def load(cls, filename):
        """Load data from file and return class object"""
        if not filename.endswith('.dat'):   # Format filename
            filename += '.dat'
        
        with open(filename, "rb") as f:
            compressed_pickle = f.read()

        pickled_data = blosc.decompress(compressed_pickle)
        data = pickle.loads(pickled_data)
        
        if type(data) is not cls:
            raise ValueError("Invalid data file.")
        return data
    
    