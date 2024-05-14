import numpy as np

def constant(learning_rate):
    """Yields a given constant learning rate"""
    while True: 
        yield learning_rate   
       
       
def step(max_lr, gamma, step_size):
    """Decaying learning rate, decaying by a parameter every step_size steps (stairs)"""
    i = 1
    while True:
        new_lr = max_lr * np.power(gamma, np.floor(i / step_size))
        i += 1
        yield new_lr
       
       
def inv(max_lr, gamma, p):
    """Decaying by an inverse parameter every step size (smooth decay)"""
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

def linear(max_lr, min_lr, num_steps):
    """Decaying linear learning rate schedule"""
    lr_diff = max_lr - min_lr
    for i in range(num_steps):
        yield max_lr - lr_diff*(i / num_steps)
    while True:
        yield min_lr

def exponential(max_lr, gamma):
    """Decaying exponential learning rate schedule"""
    i = 0
    while True:
        yield max_lr * np.exp(-gamma*i)
        i += 1
    
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
    elif (lr_type=='linear'):
        key_list = ['max_lr', 'min_lr', 'num_steps']
    elif (lr_type=='exponential'):
        key_list = ['max_lr', 'gamma']
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
    elif (lr_type=='linear'):
        return linear(**params)
    elif (lr_type=='exponential'):
        return exponential(**params)
    else:
        raise ValueError('{} is not a valid input for lr_type (type of learning rate to generate)'.format(lr_type))
