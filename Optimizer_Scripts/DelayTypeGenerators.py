# DelayTypeGenerators.py

"""Functions defining the different delay types"""

import numpy as np

class DelayType():
    def __init__(self, max_L, num_delays):
        self.max_L = max_L
        self.num_delays = num_delays
        self.delayed = True
        
    def copy(self):
        return get_delay_scheme(**self.__dict__)
        
class Undelayed(DelayType):
    def __init__(self, **kwargs):
        DelayType.__init__(self, 0, 0)
        self.delayed = False
        self.delay_type = "undelayed"
        
    def D_gen(self, n):
        while True:
            yield np.zeros(n, dtype=int)
        
class Uniform(DelayType):
    def __init__(self, max_L, num_delays, **kwargs):
        DelayType.__init__(self, max_L, num_delays)
        self.delay_type = "uniform"
        
    def D_gen(self, n):
        for i in range(self.num_delays):
            yield self.max_L*np.ones(n, dtype=int)
        while True:
            yield np.zeros(n, dtype=int)
        
class Stochastic(DelayType):
    def __init__(self, max_L, num_delays, **kwargs):
        DelayType.__init__(self, max_L, num_delays)
        self.delay_type = "stochastic"
    
    def D_gen(self, n):
        for i in range(self.num_delays):
            yield np.random.randint(0, self.max_L+1, n)
        while True:
            yield np.zeros(n, dtype=int)
    
class Decaying(DelayType):
    def __init__(self, max_L, num_delays, **kwargs):
        DelayType.__init__(self, max_L, num_delays)
        self.delay_type = "decaying"
    
    def D_gen(self, n):
        for i in range(self.num_delays):
            L = self.max_L - int(i * self.max_L / self.num_delays)
            yield L*np.ones(n, dtype=int)
        while True:
            yield np.zeros(n, dtype=int)
    
class StochDecay(DelayType):        # Unused
    def __init__(self, max_L, num_delays, **kwargs):
        DelayType.__init__(self, max_L, num_delays)
        self.delay_type = "stochastic decaying"
        
    def D_gen(self, n):
        for i in range(self.num_delays):
            L = self.max_L - int(i * self.max_L / self.num_delays)
            yield np.random.randint(0, L+1, n)
        while True:
            yield np.zeros(n, dtype=int)
    
class Partial(DelayType):
    def __init__(self, max_L, num_delays, p, **kwargs):
        DelayType.__init__(self, max_L, num_delays)
        self.p = p
        self.delay_type = "partial"
        
    def D_gen(self, n):
        for i in range(self.num_delays):
            yield self.max_L * np.random.binomial(1, self.p, n)
        while True:
            yield np.zeros(n, dtype=int)

def get_delay_scheme(delay_type, **kwargs):
    if delay_type == 'undelayed':
        return Undelayed(**kwargs)
    elif delay_type == 'uniform':
        return Uniform(**kwargs)
    elif delay_type == 'stochastic':
        return Stochastic(**kwargs)
    elif delay_type == 'decaying':
        return Decaying(**kwargs)
    elif delay_type == 'partial':
        return Partial(**kwargs)
    else:
        raise ValueError("Not a valid delay type.")
