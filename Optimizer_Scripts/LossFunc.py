# LossFunc.py

import numpy as np
from Optimizer_Scripts import functions


class LossFunc:
    """Object to hold function information."""
    
    def __init__(self, loss_name, n):
        self.loss_name = loss_name
        self.n = n
        self.initialize_function()
        
    def __eq__(self, other):
        if type(other) is type(self):
            return (self.loss_name == other.loss_name) \
                and (self.n == other.n) \
                and (self.domain == other.domain) \
                and (self.minimizer == other.minimizer)
        return False
        
    def initialize_function(self):
        """Initializes the loss and gradient functions based on loss_name. 
        Initilizes the default domain and the known minimizer for the function.
        """
        if self.loss_name.lower() == 'rosenbrock':
            self.loss = functions.rosenbrock_gen(self.n)
            self.grad = functions.rosen_deriv_gen(self.n)
            self.domain = [-5., 10.]
            self.minimizer = np.ones(self.n)
        elif self.loss_name.lower() == 'zakharov':
            self.loss = functions.zakharov_gen(self.n)
            self.grad = functions.zakharov_deriv_gen(self.n)
            self.domain = [-5., 10.]
            self.minimizer = np.zeros(self.n)
        elif self.loss_name.lower() == 'ackley':
            self.loss = functions.ackley_gen(self.n)
            self.grad = functions.ackley_deriv_gen(self.n)
            self.domain = [-32.768, 32.768]
            self.minimizer = np.zeros(self.n)
        elif self.loss_name.lower() == 'rastrigin':
            self.loss = functions.rastrigin_gen(self.n)
            self.grad = functions.rast_deriv_gen(self.n)
            self.domain = [-5.12, 5.12]
            self.minimizer = np.zeros(self.n)
        elif self.loss_name.lower() == 'stable':
            self.loss = functions.stable_gen(self.n)
            self.grad = functions.stable_deriv_gen(self.n)
            self.domain = [-3., 3.]
            self.minimizer = np.zeros(self.n)
        else:
            raise ValueError("Function name not recognized")
