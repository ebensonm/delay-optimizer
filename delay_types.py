# DelayTypes.py

"""Functions defining the different delay types"""

import numpy as np

def undelayed(n, **kwargs):
    """Returns the arguments for undelayed optimization"""
    return {'delayed':False, **kwargs}


def uniform(n, max_L=1, **kwargs):
    """Returns the arguments for uniformly delayed optimization"""
    D = [max_L * np.ones(n, dtype=int)]
    return {'delayed':True, 'D':D, 'random':False, **kwargs}


def stochastic(n, **kwargs):
    """Returns the arguments for stochastically delayed optimization"""
    return {'delayed':True, 'D':None, 'random':True, **kwargs}


def decaying(n, max_L=2, num_delays=1000, stochastic=False, **kwargs):
    """Returns the arguments for decaying delayed optimization.
        stochastic (bool): whether to perform stochastic or uniform delays
                           on the decaying max_L
                           
    *Basically guarantees that delays are smaller the more iters have passed*
    """
    # Get array of decaying max_Ls
    L = [max_L - int(t*max_L/num_delays) for t in range(num_delays)]
    
    # Define delays
    if stochastic is True:
        D = [np.random.randint(0,l+1,size=n) for l in L]
    else:
        D = [l*np.ones(n, dtype=int) for l in L]
        
    kwargs['num_delays'] = num_delays
    
    return {'delayed':True, 'D':D, 'random':False, **kwargs}


def partial(n, max_L=1, num_delays=1000, stochastic=False, p=0.5, **kwargs):
    """Returns the arguments for partial delayed optimization.
        p (float): percent of the dimensions to delay
        
    *Basically guarantees that p percent of dimensions are delayed each iter*
    """
    # Determine how many coordinates are delayed
    d = max(int(n*p),1)
    
    # Choose which dimensions to delay at each time step
    dims = [np.random.choice(np.arange(0,n), replace=False, size=d) 
            for i in range(num_delays)]
    
    # Define delays
    D = [np.zeros(n, dtype=int) for i in range(num_delays)]
    if stochastic is True:
        for i in range(num_delays):
            D[i][dims[i]] = np.random.randint(1, max_L+1, size=d)
    else:
        for i in range(num_delays):
            D[i][dims[i]] = max_L
            
    kwargs['num_delays'] = num_delays
    
    return {'delayed':True, 'D':D, 'random':False, **kwargs}


def parse_delay_type(type_str, n, **kwargs):
    if type_str == 'undelayed':
        return undelayed(n, **kwargs)
    elif type_str == 'uniform':
        return uniform(n, **kwargs)
    elif type_str == 'stochastic':
        return stochastic(n, **kwargs)
    elif type_str == 'decaying':
        return decaying(n, **kwargs)
    elif type_str == 'partial':
        return partial(n, **kwargs)
    else:
        raise ValueError("Not a valid delay type.")
