import numpy as np
from ..generators import learning_rate_generator as lr_gen

class Optimizer:
    def __init__(self, lr=0.01):
        if isinstance(lr, (float, int)):
            lr = lr_gen.constant(lr)
        self.lr = lr
        self.initialized = False

    def initialize(self, x_init):
        raise NotImplementedError("Subclasses must implement this method")

    def step(self, x, grad):
        raise NotImplementedError("Subclasses must implement this method")

    def __call__(self):
        return self.step()
       

class GradientDescent(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__(lr)

    def initialize(self, x_init):
        self.initialized = True

    def step(self, x, grad):
        return x - next(self.lr) * grad

      
class Adam(Optimizer):
    def __init__(self, lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        super().__init__(lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def initialize(self, x_init):
        self.n = x_init.shape[-1]
        self.m = np.zeros(self.n)
        self.v = np.zeros(self.n)
        self._t = 0
        self.initialized = True
               
    def step(self, x, grad):
        self._t += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.square(grad)
        m_hat = self.m / (1 - self.beta_1 ** self._t)
        v_hat = self.v / (1 - self.beta_2 ** self._t)
        return x - (next(self.lr) *  m_hat) / (np.sqrt(v_hat) + self.epsilon)
        

# TODO: Check this against an algorithm
class Momentum(Optimizer):
    def __init__(self, lr=0.01, gamma=0.9):
        super().__init__(lr)
        self.gamma = gamma
        
    def initialize(self, x_init):
        self.n = x_init.shape[-1]
        self.v = np.zeros(self.n, dtype=int)    # Should this be integers?
        self.initialized = True
    
    def step(self, x, grad):
        self.v = self.gamma * self.v + next(self.lr) * grad  # Is this right for v?
        return x - self.v
            

# TODO: Check this against an algorithm
class NesterovMomentum(Optimizer):
    def __init__(self, lr=0.01, gamma=0.9):
        super().__init__(lr)
        self.gamma = gamma
    
    def initialize(self, x_init):
        self.n = len(x_init)
        self.v = np.zeros(self.n)
        self.grad_helper = np.zeros(self.n)
        self.initialized = True
        
    def step(self, x, grad):
        self.v = self.gamma * self.v + next(self.lr) * grad
        self.grad_helper = self.gamma * self.v  # What is the point of this?
        return x - self.v                       # Same as the normal momentum, what is the difference?

    
