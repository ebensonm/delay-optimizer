import numpy as np

def constant(params):
    """Yields a given constant learning rate
    """
    GO = True
    learning_rate = params['learning_rate']
    while GO is True:
       yield learning_rate     
       
       
def step(params):
    """Decaying learning rate, decaying by a parameter every step_size steps (stairs)
    """
    GO = True
    step_size=params['step_size']
    lr = params['max_lr']
    gamma = params['gamma']
    old_lr = lr
    iternum = 1
    while GO is True:
        new_lr = old_lr*np.power(gamma,np.floor(iternum/step_size))
        iternum += 1
        yield new_lr
       
       
def inv(params):
    """Decaying by an inverse parameter every step size (smooth decay)
    """
    GO = True
    gamma =params['gamma']
    lr = params['max_lr']
    p = params['p']
    old_lr = lr
    iternum = 1
    while GO is True:
        new_lr = old_lr*np.power(1/(1+iternum*gamma),p) 
        iternum += 1
        yield new_lr
        

def tri_2(params):
    """Decaying triangle learning rate schedule"""
    GO = True
    max_lr = params['max_lr']
    min_lr = params['min_lr']
    step_size = params['step_size']
    old_lr = max_lr
    iternum = 1
    while GO is True:
        value = 2/np.pi*np.absolute(np.arcsin(np.sin(np.pi*iternum/(2*step_size))))
        new_lr = (1/np.power(2,np.floor(iternum/(2*step_size))))*np.absolute(max_lr-min_lr)*value+np.min((max_lr, min_lr))
        iternum += 1
        yield new_lr
    
    
def sin_2(params):
    """Decaying sin learning rate schedule"""
    GO = True
    max_lr = params['max_lr']
    min_lr = params['min_lr']
    step_size = params['step_size']
    old_lr = max_lr
    iternum = 1
    while GO is True:
        value = np.absolute(np.sin(np.pi*iternum/(2*step_size)))
        new_lr = (1/np.power(2,np.floor(iternum/(2*step_size))))*np.absolute(max_lr-min_lr)*value+np.min((max_lr, min_lr))
        iternum += 1
        yield new_lr
       
    
def get_param_dict(lr_type):
    param = dict()
    if (lr_type == 'step'):
        key_list = ['max_lr', 'gamma', 'step_size'] 
    
    elif (lr_type =='inv'):
        key_list = ['max_lr', 'p', 'gamma']
    
    elif (lr_type=='tri-2'):
        key_list = ['max_lr', 'min_lr', 'step_size']
    
    elif (lr_type=='sin-2'):
        key_list = ['max_lr', 'min_lr', 'step_size']
    
    else:
        raise ValueError("Not a valid lr_type") 
    for i in key_list:
        param[i] = None
    return param    
        
def generate_learning_rates(lr_type, params):
    """ Create the learning rate generator for constant and nonconstant learning rates
    """
    if (lr_type == 'const'):
        return constant(params)
    elif (lr_type == 'step'):
        return step(params)
    elif (lr_type == 'inv'):
        return inv(params)
    elif (lr_type == 'tri-2'):
        return tri_2(params)
    elif (lr_type == 'sin-2'):
        return sin_2(params)
    else:
        raise ValueError('{} is not a valid input for lr_type (type of learning rate to generate)'.format(lr_type))
