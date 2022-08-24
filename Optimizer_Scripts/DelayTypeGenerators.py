# DelayTypeGenerators.py

"""Functions defining the different delay types"""

import numpy as np

class DelayType():
    def __init__(self, max_L, num_delays):
        self.max_L = max_L
        self.num_delays = num_delays
        self.delayed = True
        
class Undelayed(DelayType):
    def __init__(self):
        DelayType.__init__(self, 0, 0)
        self.delayed = False
        self.name = "undelayed"
        
    def D_gen(self, n):
        while True:
            yield np.zeros(n, dtype=int)
        
class Uniform(DelayType):
    def __init__(self, max_L, num_delays):
        DelayType.__init__(self, max_L, num_delays)
        self.name = "uniform"
        
    def D_gen(self, n):
        for i in range(self.num_delays):
            yield self.max_L*np.ones(n, dtype=int)
        while True:
            yield np.zeros(n, dtype=int)
        
class Stochastic(DelayType):
    def __init__(self, max_L, num_delays):
        DelayType.__init__(self, max_L, num_delays)
        self.name = "stochastic"
    
    def D_gen(self, n):
        for i in range(self.num_delays):
            yield np.random.randint(0, self.max_L+1, n)
        while True:
            yield np.zeros(n, dtype=int)
    
class Decaying(DelayType):
    def __init__(self, max_L, num_delays):
        DelayType.__init__(self, max_L, num_delays)
        self.name = "decaying"
    
    def D_gen(self, n):
        for i in range(self.num_delays):
            L = self.max_L - int(i * self.max_L / self.num_delays)
            yield L*np.ones(n, dtype=int)
        while True:
            yield np.zeros(n, dtype=int)
    
class StochDecay(DelayType):
    def __init__(self, max_L, num_delays):
        DelayType.__init__(self, max_L, num_delays)
        self.name = "stochastic decaying"
        
    def D_gen(self, n):
        for i in range(self.num_delays):
            L = self.max_L - int(i * self.max_L / self.num_delays)
            yield np.random.randint(0, L+1, n)
        while True:
            yield np.zeros(n, dtype=int)
    
class Partial(DelayType):
    def __init__(self, max_L, num_delays, p):
        DelayType.__init__(self, max_L, num_delays)
        self.p = p
        self.name = "partial"
        
    def D_gen(self, n):
        for i in range(self.num_delays):
            yield self.max_L * np.random.binomial(1, self.p, n)
        while True:
            yield np.zeros(n, dtype=int)

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
