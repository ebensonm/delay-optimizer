# delay_types.py

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


def decaying(n, max_L=1, maxiter=2000, stochastic=False, **kwargs):
    """Returns the arguments for decaying delayed optimization.
        stochastic (bool): whether to perform stochastic or uniform delays
                           on the decaying max_L
                           
    *Basically guarantees that delays are smaller the more iters have passed*
    """
    # Get array of decaying max_Ls
    m = maxiter / (max_L+1)
    L = np.repeat(np.arange(max_L+1)[::-1], m)
    L = np.pad(L, (0,maxiter-len(L)))
    
    # Define delays
    if stochastic is True:
        D = [np.random.randint(0,l+1,size=n) for l in L]
    else:
        D = [l*np.ones(n, dtype=int) for l in L]
        
    kwargs['maxiter'] = maxiter
    kwargs['num_delays'] = maxiter
    
    return {'delayed':True, 'D':D, 'random':False, **kwargs}


def partial(n, max_L=1, maxiter=2000, stochastic=False, p=0.5, **kwargs):
    """Returns the arguments for partial delayed optimization.
        p (float): percent of the dimensions to delay
        
    *Basically guarantees that p percent of dimensions are delayed each iter*
    """
    # Determine how many dimensions are delayed
    d = int(n*p)
    if d == 0: 
        d = 1
    
    # Choose which dimensions to delay at each time step
    dims = [np.random.choice(np.arange(0,n), replace=False, size=d) 
            for i in range(maxiter)]
    
    # Define delays
    D = [np.zeros(n, dtype=int) for i in range(maxiter)]
    if stochastic is True:
        for i in range(maxiter):
            D[i][dims[i]] = np.random.randint(1, max_L+1, size=d)
    else:
        for i in range(maxiter):
            D[i][dims[i]] = max_L
    
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
