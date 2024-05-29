# DelayTypeGenerators.py

import numpy as np

# TODO: Should I find a way to make the classes themselves iterables?

class DelayType():
    def __init__(self, delay_type, max_L, num_delays):
        self.max_L = max_L
        self.num_delays = num_delays

    def __repr__(self):
        return f"{self.__class__.__name__}(max_L={self.max_L}, num_delays={self.num_delays})"

    def __str__(self):
        return self.__class__.__name__

        
class Undelayed(DelayType):
    def __init__(self):
        super().__init__(max_L=0, num_delays=0)
        
    def D_gen(self, n):
        while True:
            yield np.zeros(n, dtype=int)

        
class Uniform(DelayType):
    def D_gen(self, n):
        for i in range(self.num_delays):
            yield self.max_L*np.ones(n, dtype=int)
        while True:
            yield np.zeros(n, dtype=int)

        
class Stochastic(DelayType):
    def D_gen(self, n):
        for i in range(self.num_delays):
            yield np.random.randint(0, self.max_L+1, n)
        while True:
            yield np.zeros(n, dtype=int)
    

class Decaying(DelayType):
    def D_gen(self, n):
        for i in range(self.num_delays):
            L = self.max_L - int(i * self.max_L / self.num_delays)
            yield L*np.ones(n, dtype=int)
        while True:
            yield np.zeros(n, dtype=int)

    
class Partial(DelayType):
    def __init__(self, max_L, num_delays, p):
        super().__init__(max_L, num_delays)
        self.p = p
        
    def D_gen(self, n):
        for i in range(self.num_delays):
            yield self.max_L * np.random.binomial(1, self.p, n)
        while True:
            yield np.zeros(n, dtype=int)

