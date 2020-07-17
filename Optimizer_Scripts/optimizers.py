import numpy as np
import time

class Adam:

    def __init__(self, params, epsilon=1e-7):
        self.params = params
        self.epsilon = epsilon
        self.name = 'Adam'    
        self.initialized = False 
          
    def initialize(self,x_init):
        self.n = len(x_init)
        self.x_state = x_init
        self.m_t = np.zeros(self.n)
        self.v_t = np.zeros(self.n)
        self.grad_helper = np.zeros(self.n) 
        self.initialized = True
               
    def __call__(self, x_state, x_grad, iteration_num):
        #update parameters
        self.m_t = self.params['beta_1'] * self.m_t + (1-self.params['beta_1']) * x_grad
        self.v_t = self.params['beta_2'] * self.v_t + (1-self.params['beta_2']) * np.power(x_grad,2)
        m_t_hat = self.m_t / (1 - np.power(self.params['beta_1'],iteration_num))
        v_t_hat = self.v_t / (1 - np.power(self.params['beta_2'],iteration_num))
        self.x_state = x_state - (self.params['learning_rate'][iteration_num -1] *  m_t_hat) / (np.sqrt(v_t_hat) + self.epsilon)
        return self.x_state
        
class Momentum:

    def __init__(self, params):
        self.params = params
        self.name = 'Momentum'
        self.initialized = False
        
    def initialize(self, x_init):
        self.n = len(x_init)
        self.x_state = x_init
        self.v_k = np.zeros(self.n,dtype=int)
        self.grad_helper = np.zeros(self.n,dtype=int)
        self.initialized = True
    
    def __call__(self, x_state, x_grad, iteration_num):
        #update parameters
        self.v_k = self.params['gamma'] * self.v_k + self.params['learning_rate'][iteration_num-1] * x_grad
        self.x_state = x_state - self.v_k
        return self.x_state
            
class NesterovMomentum:

    def __init__(self, params):
        self.params = params
        self.name = "Nesterov Momentum"
        self.initialized = False
    
    def initialize(self, x_init):
        self.n = len(x_init)
        self.x_state = x_init
        self.v_k = np.zeros(self.n)
        self.grad_helper = self.v_k
        self.initialized = True
        
    def __call__(self, x_state, x_grad, iteration_num):
        #update parameters
        self.v_k = self.params['gamma'] * self.v_k + self.params['learning_rate'][iteration_num] * x_grad
        self.x_state = x_state - self.v_k
        self.grad_helper = self.params['gamma'] * self.v_k
        return self.x_state
    
