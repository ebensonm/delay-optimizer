import numpy as np
from typing import Generator, List, Union

class DelayType():
    def __init__(self, max_L: int, num_delays: int):
        self.max_L = max_L
        self.num_delays = num_delays

    def D_gen(self, size) -> Generator[np.ndarray[int], None, None]:
        raise NotImplementedError("Subclasses must implement D_gen method")

    def __repr__(self):
        return f"{self.__class__.__name__}(max_L={self.max_L}, num_delays={self.num_delays})"

    def __str__(self):
        return self.__class__.__name__

        
class Undelayed(DelayType):
    def __init__(self):
        super().__init__(max_L=0, num_delays=0)
        
    def D_gen(self, size) -> Generator[np.ndarray[int], None, None]:
        while True:
            yield np.zeros(size, dtype=int)

        
class Uniform(DelayType):
    def D_gen(self, size) -> Generator[np.ndarray[int], None, None]:
        for i in range(self.num_delays):
            yield np.full(size, self.max_L, dtype=int)
        while True:
            yield np.zeros(size, dtype=int)

        
class Stochastic(DelayType):
    def D_gen(self, size) -> Generator[np.ndarray[int], None, None]:
        for i in range(self.num_delays):
            yield np.random.randint(0, self.max_L+1, size=size)
        while True:
            yield np.zeros(size, dtype=int)
    

class Decaying(DelayType):
    def D_gen(self, size) -> Generator[np.ndarray[int], None, None]:
        for i in range(self.num_delays):
            L = self.max_L - int(i * self.max_L / self.num_delays)
            yield np.full(size, L, dtype=int)
        while True:
            yield np.zeros(size, dtype=int)

    
class Partial(DelayType):
    def __init__(self, max_L: int, num_delays: int, p: float):
        super().__init__(max_L, num_delays)
        self.p = p
        
    def D_gen(self, size) -> Generator[np.ndarray[int], None, None]:
        for i in range(self.num_delays):
            yield self.max_L * np.random.binomial(1, self.p, size=size)
        while True:
            yield np.zeros(size, dtype=int)


class Cyclical(DelayType):
    def __init__(self, D: List[np.ndarray[int]], num_delays: int):
        super().__init__(max_L=np.max(D), num_delays=num_delays)
        D = np.atleast_2d(D)
        if (D < 0).any():
            raise ValueError("Delay distribution D can only contain non-negative integers")
        self.D = D

    def D_gen(self, size) -> Generator[np.ndarray[int], None, None]:
        size = np.atleast_1d(size)
        if size[-1] != self.D.shape[-1]:
            raise ValueError("Delay distribution vector D does not match the input size")
        for i in range(self.num_delays):
            yield np.tile(self.D[i % len(self.D)], (*size[:-1], 1))
        while True:
            yield np.zeros(size, dtype=int)


class Constant(Cyclical):
    def __init__(self, D: np.ndarray[int], num_delays: int):
        super().__init__(D=[D], num_delays=num_delays)

