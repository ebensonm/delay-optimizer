# FuncOptHandler.py 

import numpy as np
import warnings
from Optimizer_Scripts.Data import Data
from tqdm import tqdm

from ..optimization import functions, optimizers, schedulers
from ..delays import DelayedOptimizer, distributions
from .parse import (
    parse_objective_function,
    parse_scheduler,
    parse_optimizer,
    parse_delay_distribution
)

class Handler:
    """Class for handling delayed or undelayed optimization on a given function"""
    def __init__(self, objective, **kwargs):
        """Initializer for the Handler class
        
        Parameters: 
            objective(str): name of the loss function to be analyzed
            dim(int): the dimension of the state vector
        """
        self.objective = parse_objective_function(objective, **kwargs)

    # Initialize points -----------------------------------------------------

    def random_points(self, num_points):
        """Randomly initialize given number of points within the domain of the objective function"""
        self.x_inits = np.random.uniform(
            *self.objective.domain, 
            size=(num_points, self.objective.n)
        )
        
    def load_points(self, points):
        """Load initial optimization points into the Handler object"""
        points = np.atleast_2d(points)
        if points.shape[1] != self.objective.n:
            raise ValueError(
                "Points array does not match function dimension. Please provide "
                f"an array of points with shape (*,{self.objective.n}).")
        self.x_inits = points
    

    # Run optimization ------------------------------------------------------
   
    def optimize(self, optimizer, delays, scheduler="constant", maxiter=5000, 
                 save_state=(0,1), save_loss=True, save_grad=False, **kwargs):      # TODO: TEST THIS
        """Run the optimization on the initial points already initialized and 
        saves values to be plotted.
        
        Parameters:
            optimizer(Optimizer,str): the base optimizer
            delays(DelayType,str): the delay distribution to apply during optimization
            scheduler(Scheduler,str): the learning rate scheduler to use
            maxiter(int): the maximum number of iterations for optimization
            save_state(bool/tuple): state dimensions to save during optimization
            save_loss(bool): whether to save loss values over time 
            save_grad(bool): whether to save gradient values over time
        """
        # Check if points have been initialized
        if len(self.x_inits) == 0:
            warnings.warn("No points have been initialized.")
            return
        
        # Initialize
        scheduler_kwargs, optimizer_kwargs, delay_kwargs = parse_kwargs(kwargs)
        scheduler = parse_scheduler(scheduler, **scheduler_kwargs)
        optimizer = parse_optimizer(optimizer, lr=scheduler, **optimizer_kwargs)
        delays = parse_delay_distribution(delays, **delay_kwargs)
        delayer = DelayedOptimizer(self.objective, optimizer, delays)

        # self.data.set_delay_scheme(delay_type, maxiter, tol, break_opt)
        # self.data.set_optimizer_params(optimizer_name, lr_params)
            
        pbar = tqdm(
            total=maxiter,
            desc=r"{} {}d ({})".format(self.objective.__class__.__name__, 
                                        self.objective.n,
                                        delay_type.__class__.__name__), 
            leave=True
        )
        self.initialize(x_init)
        for i in pbar:
            delayer.step()
            pbar.update()
            
        return


    # Save and load data ----------------------------------------------------

    def save_data(self, filename):
        self.data.save(filename)
        
    def load_data(self, filename):
        data = Data.load(filename)
        if (self.objective != data.get_loss_function()):
            raise ValueError("Functions do not match. Data was not loaded.")
        self.data = data
        
    @classmethod
    def load(cls, filename):
        """Load Handler object from Data class file"""
        data = Data.load(filename)
        obj = cls(data.loss_name, data.dim)
        obj.x_inits = data.get_x_inits()
        obj.data = data
        return obj
    

    # Data management -----------------------------------------------------
    
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
        
    
        
