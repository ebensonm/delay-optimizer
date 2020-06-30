import numpy as np

class Adam:

    def __init__(self, learning_rate=1.05, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.name = 'Adam'    
        self.initialized = False 
          
    def initialize(self,x_init):
        self.n = len(x_init)
        self.x_state = x_init
        self.m_t = np.zeros(self.n)
        self.v_t = np.zeros(self.n) 
               
    def __call__(self, x_state, x_grad, iteration_num):
        if (not self.initialized):
            self.initialize(x_state)
            self.initialized = True
        #update parameters
        self.m_t = self.beta_1 * self.m_t + (1-self.beta_1) * x_grad
        self.v_t = self.beta_2 * self.v_t + (1-self.beta_2) * np.power(x_grad,2)
        m_t_hat = self.m_t / (1 - np.power(self.beta_1,iteration_num))
        v_t_hat = self.v_t / (1 - np.power(self.beta_2,iteration_num))
        self.x_state = x_state - (self.learning_rate *  m_t_hat) / (np.sqrt(v_t_hat) + self.epsilon)
        return self.x_state
