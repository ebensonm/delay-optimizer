import numpy as np


class Optimizer:
    def __init__(self, lr=0.01):
        if isinstance(lr, (float, int)):
            lr = constant(lr)
        self.lr = lr
        self.initialized = False

    def initialize(self, x_init):
        raise NotImplementedError("Subclasses must implement this method")

    def step(self, x, grad):
        raise NotImplementedError("Subclasses must implement this method")

    def __call__(self, x, grad):
        return self.step(x, grad)
       

class GradientDescent(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__(lr)

    def initialize(self, x_init):
        self.initialized = True

    def step(self, x, grad):
        return x - next(self.lr) * grad

      
class Adam(Optimizer):
    def __init__(self, lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        super().__init__(lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def initialize(self, x_init):
        self.n = x_init.shape[-1]
        self.m = np.zeros(self.n)
        self.v = np.zeros(self.n)
        self._t = 0
        self.initialized = True
               
    def step(self, x, grad):
        self._t += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.square(grad)
        m_hat = self.m / (1 - self.beta_1 ** self._t)
        v_hat = self.v / (1 - self.beta_2 ** self._t)
        return x - (next(self.lr) *  m_hat) / (np.sqrt(v_hat) + self.epsilon)
        

# TODO: Check this against an algorithm
class Momentum(Optimizer):
    def __init__(self, lr=0.01, gamma=0.9):
        super().__init__(lr)
        self.gamma = gamma
        
    def initialize(self, x_init):
        self.n = x_init.shape[-1]
        self.v = np.zeros(self.n, dtype=int)    # Should this be integers?
        self.initialized = True
    
    def step(self, x, grad):
        self.v = self.gamma * self.v + next(self.lr) * grad  # Is this right for v?
        return x - self.v
            

# TODO: Check this against an algorithm
class NesterovMomentum(Optimizer):
    def __init__(self, lr=0.01, gamma=0.9):
        super().__init__(lr)
        self.gamma = gamma
    
    def initialize(self, x_init):
        self.n = len(x_init)
        self.v = np.zeros(self.n)
        self.grad_helper = np.zeros(self.n)
        self.initialized = True
        
    def step(self, x, grad):
        self.v = self.gamma * self.v + next(self.lr) * grad
        self.grad_helper = self.gamma * self.v  # What is the point of this?
        return x - self.v                       # Same as the normal momentum, what is the difference?


def constant(learning_rate):
    """Yields a given constant learning rate
    """
    while True: yield learning_rate   
       
       
def step(max_lr, gamma, step_size):
    """Decaying learning rate, decaying by a parameter every step_size steps (stairs)
    """
    i = 1
    while True:
        new_lr = max_lr * np.power(gamma, np.floor(i / step_size))
        i += 1
        yield new_lr
       
       
def inv(max_lr, gamma, p):
    """Decaying by an inverse parameter every step size (smooth decay)
    """
    i = 1
    while True:
        new_lr = max_lr * np.power(1 / (1 + i*gamma), p) 
        i += 1
        yield new_lr
        

def tri_2(max_lr, min_lr, step_size):
    """Decaying triangle learning rate schedule"""
    i = 1
    while True:
        val1 = i / (2 * step_size)
        val2 = 2 / np.pi * np.abs(np.arcsin(np.sin(np.pi * val1)))
        val3 = np.abs(max_lr - min_lr) / np.power(2, np.floor(val1)) 
        new_lr = val2 * val3 + np.min((max_lr, min_lr))
        
        i += 1
        yield new_lr
    
    
def sin_2(max_lr, min_lr, step_size):
    """Decaying sin learning rate schedule"""
    i = 1
    while True:
        val1 = i / (2 * step_size)
        val2 = np.abs(np.sin(np.pi * val1))
        val3 = np.abs(max_lr - min_lr) / np.power(2, np.floor(val1))
        new_lr = val2 * val3 + np.min((max_lr, min_lr))
        
        i += 1
        yield new_lr
       
    
def get_param_dict(lr_type):
    if (lr_type == 'const'):
        key_list = ['learning_rate']
    elif (lr_type == 'step'):
        key_list = ['max_lr', 'gamma', 'step_size'] 
    elif (lr_type =='inv'):
        key_list = ['max_lr', 'gamma', 'p']
    elif (lr_type=='tri-2'):
        key_list = ['max_lr', 'min_lr', 'step_size']
    elif (lr_type=='sin-2'):
        key_list = ['max_lr', 'min_lr', 'step_size']
    else:
        raise ValueError("Not a valid lr_type") 
        
    params = dict()
    for key in key_list:
        params[key] = None
        
    return params    
        
def generate_learning_rates(lr_type, **params):
    """ Create the learning rate generator for constant and nonconstant learning rates
    """
    if (lr_type == 'const'):
        return constant(**params)
    elif (lr_type == 'step'):
        return step(**params)
    elif (lr_type == 'inv'):
        return inv(**params)
    elif (lr_type == 'tri-2'):
        return tri_2(**params)
    elif (lr_type == 'sin-2'):
        return sin_2(**params)
    else:
        raise ValueError('{} is not a valid input for lr_type (type of learning rate to generate)'.format(lr_type))
