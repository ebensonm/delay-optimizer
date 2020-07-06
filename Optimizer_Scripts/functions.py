from autograd import numpy as np

def ackley_gen(n):
    def ackley(x,*args):  
        return -20 * np.exp(-0.2 * np.sqrt((1/n) * np.sum(np.square(x)))) - np.exp((1/n)*np.sum(np.cos(2*np.pi*x))) + 20 + np.exp(1)
    return ackley
    
def ackley_deriv_gen(n):
    def ackley_grad(x,*args):
        if (np.allclose(np.zeros(n), x,atol=1e-20)):
            return np.zeros(n,dtype=float)
        else:
            part_1 = 4 * x / np.sqrt(n * np.sum(np.square(x)))
            part_1_const = np.exp(-0.2 * np.sqrt(1/n * np.sum(np.square(x))))
            part_2 = (2*np.pi/n * np.sin(2*np.pi*x))
            part_2_const = np.exp((1/n) * np.sum(np.cos(2*np.pi*x)))
            return part_1 * part_1_const + part_2 * part_2_const
        
    return ackley_grad

def rastrigin_gen(n):
    def rastrigin(x,*args):
        return 10 * n + np.sum(np.square(x) - 10 * np.cos(2*np.pi*x))
    return rastrigin

def rast_deriv_gen(n):
    def rast_grad(x,*args):
        return 2 * x + 20 * np.pi * np.sin(2*np.pi*x)        
    return rast_grad
