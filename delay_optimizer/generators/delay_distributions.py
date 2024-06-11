# DelayTypeGenerators.py

import numpy as np
from typing import Generator


class DelayType():
    def __init__(self, delay_type, max_L, num_delays):
        self.max_L = max_L
        self.num_delays = num_delays

    def D_gen(self, n: int) -> Generator[np.ndarray[int], None, None]:
        raise NotImplementedError("Subclasses must implement D_gen method")

    def __repr__(self):
        return f"{self.__class__.__name__}(max_L={self.max_L}, num_delays={self.num_delays})"

    def __str__(self):
        return self.__class__.__name__

        
class Undelayed(DelayType):
    def __init__(self):
        super().__init__(max_L=0, num_delays=0)
        
    def D_gen(self, n: int) -> Generator[np.ndarray[int], None, None]:
        while True:
            yield np.zeros(n, dtype=int)

        
class Uniform(DelayType):
    def D_gen(self, n: int) -> Generator[np.ndarray[int], None, None]:
        for i in range(self.num_delays):
            yield self.max_L*np.ones(n, dtype=int)
        while True:
            yield np.zeros(n, dtype=int)

        
class Stochastic(DelayType):
    def D_gen(self, n: int) -> Generator[np.ndarray[int], None, None]:
        for i in range(self.num_delays):
            yield np.random.randint(0, self.max_L+1, n)
        while True:
            yield np.zeros(n, dtype=int)
    

class Decaying(DelayType):
    def D_gen(self, n: int) -> Generator[np.ndarray[int], None, None]:
        for i in range(self.num_delays):
            L = self.max_L - int(i * self.max_L / self.num_delays)
            yield L*np.ones(n, dtype=int)
        while True:
            yield np.zeros(n, dtype=int)

    
class Partial(DelayType):
    def __init__(self, max_L: int, num_delays: int, p: float):
        super().__init__(max_L, num_delays)
        self.p = p
        
    def D_gen(self, n: int) -> Generator[np.ndarray[int], None, None]:
        for i in range(self.num_delays):
            yield self.max_L * np.random.binomial(1, self.p, n)
        while True:
            yield np.zeros(n, dtype=int)


class Cyclical(DelayType):
    def __init__(self, D: list[np.ndarray[int]], num_delays: int):
        super().__init__(max_L=np.max(D), num_delays=num_delays)
        if any([dt < 0 for dt in D]):
            raise ValueError("Delay distribution D can only contain non-negative integers")
        self.D = D

    def D_gen(self, n: int) -> Generator[np.ndarray[int], None, None]:
        if n != len(self.D[0]):
            raise ValueError("Delay distribution vector D does not match the input dimensionality n")
        for i in range(self.num_delays):
            yield self.D[i % len(self.D)]
        while True:
            yield np.zeros(n, dtype=int)


class Constant(DelayType):
    def __init__(self, D: np.ndarray[int], num_delays: int):
        super().__init__(D=[D], num_delays=num_delays)

