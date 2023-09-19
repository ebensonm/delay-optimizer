# DelayTypeGenerators.py

"""Classes defining generators for the different delay types"""

import numpy as np

class DelayType():
    def __init__(self, delay_type, max_L, num_delays, delayed=True):
        self.delay_type = delay_type
        self.max_L = max_L
        self.num_delays = num_delays
        self.delayed = delayed
        
    def copy(self):
        """Return a copy of the DelayType object.
        The state of the delay generator is not maintained in the copy.
        """
        return get_delay_scheme(**self.__dict__)
        
class Undelayed(DelayType):
    def __init__(self, **kwargs):
        DelayType.__init__(self, "undelayed", max_L=0, num_delays=0, delayed=False)
        
    def D_gen(self, n):
        while True:
            yield np.zeros(n, dtype=int)
        
class Uniform(DelayType):
    def __init__(self, max_L, num_delays, **kwargs):
        DelayType.__init__(self, "uniform", max_L, num_delays)
        
    def D_gen(self, n):
        for i in range(self.num_delays):
            yield self.max_L*np.ones(n, dtype=int)
        while True:
            yield np.zeros(n, dtype=int)
        
class Stochastic(DelayType):
    def __init__(self, max_L, num_delays, **kwargs):
        DelayType.__init__(self, "stochastic", max_L, num_delays)
    
    def D_gen(self, n):
        for i in range(self.num_delays):
            yield np.random.randint(0, self.max_L+1, n)
        while True:
            yield np.zeros(n, dtype=int)
    
class Decaying(DelayType):
    def __init__(self, max_L, num_delays, **kwargs):
        DelayType.__init__(self, "decaying", max_L, num_delays)
    
    def D_gen(self, n):
        for i in range(self.num_delays):
            L = self.max_L - int(i * self.max_L / self.num_delays)
            yield L*np.ones(n, dtype=int)
        while True:
            yield np.zeros(n, dtype=int)
    
class StochDecay(DelayType):        # Unused
    def __init__(self, max_L, num_delays, **kwargs):
        DelayType.__init__(self, "stochastic decaying", max_L, num_delays)
        
    def D_gen(self, n):
        for i in range(self.num_delays):
            L = self.max_L - int(i * self.max_L / self.num_delays)
            yield np.random.randint(0, L+1, n)
        while True:
            yield np.zeros(n, dtype=int)
    
class Partial(DelayType):
    def __init__(self, max_L, num_delays, p, **kwargs):
        DelayType.__init__(self, "partial", max_L, num_delays)
        self.p = p
        
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
