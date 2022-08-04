# DelayTypes.py

"""Functions defining the different delay types"""

import numpy as np

class DelayType():
    def __init__(self, max_L, num_delays):
        self.delayed = True
        self.max_L = max_L
        self.num_delays = num_delays
        self.random = False
        
    # TODO: Maybe a parse function for getting delay parameters (D, random, etc)
        
        
class Undelayed(DelayType):
    def __init__(self):
        DelayType.__init__(self, 0, 0)
        self.delayed = False
        self.name = "undelayed"
        
    def D(self, n):
        return []
        
class Uniform(DelayType):
    def __init__(self, max_L, num_delays):
        DelayType.__init__(self, max_L, num_delays)
        self.name = "uniform"
        
    def D(self, n):
        return [self.max_L * np.ones(n, dtype=int)]
        
class Stochastic(DelayType):
    def __init__(self, max_L, num_delays):
        DelayType.__init__(self, max_L, num_delays)
        self.random = True
        self.name = "stochastic"
    
    def D(self, n):
        return []
    
class Decaying(DelayType):
    def __init__(self, max_L, num_delays):
        DelayType.__init__(self, max_L, num_delays)
        self.name = "decaying"
    
    def D(self, n):
        L = [self.max_L - int(t * self.max_L / self.num_delays) 
             for t in range(self.num_delays)]
        return [l * np.ones(n, dtype=int) for l in L]
    
class StochDecay(DelayType):
    def __init__(self, max_L, num_delays):
        DelayType.__init__(self, max_L, num_delays)
        self.name = "stochastic decaying"
        
    def D(self, n):
        L = [self.max_L - int(t * self.max_L / self.num_delays) 
             for t in range(self.num_delays)]
        return [np.random.randint(l+1, size=n, dtype=int) for l in L]
    
class Partial(DelayType):
    def __init__(self, max_L, num_delays, p):
        DelayType.__init__(self, max_L, num_delays)
        self.p = p
        self.name = "partial"
    
    def D(self, n):
        d = max(int(n * self.p),1)                # Number of coords to delay
        dims = [np.random.choice(np.arange(0,n), replace=False, size=d) 
                for i in range(self.num_delays)]  # Which coords to delay
        
        D = [np.zeros(n, dtype=int) for i in range(self.num_delays)]
        for i in range(self.num_delays):
            D[i][dims[i]] = self.max_L
            
        return D

def gen_delay_type(type_str, **kwargs):
    if type_str == 'undelayed':
        return Undelayed(**kwargs)
    elif type_str == 'uniform':
        return Uniform(**kwargs)
    elif type_str == 'stochastic':
        return Stochastic(**kwargs)
    elif type_str == 'decaying':
        return Decaying(**kwargs)
    elif type_str == 'partial':
        return Partial(**kwargs)
    else:
        raise ValueError("Not a valid delay type.")
