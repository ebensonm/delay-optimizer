import numpy as np


def adaptive_function(grad, x_value, max_delay):
    #get the new time delay value, for now, just take the max of the two values
    time_del = max(int(np.ceil(1/np.linalg.norm(grad(x_value)))), max_delay)
    return time_del

def ackley_gen(n):
    def ackley(x,*args):  
        return -20 * np.exp(-0.2 * np.sqrt((1/n) * np.sum(np.square(x)))) - np.exp((1/n)*np.sum(np.cos(2*np.pi*x))) + 20 + np.exp(1)
    return ackley
    
def ackley_deriv_gen(n):
    def ackley_grad(x,*args):
        part_1 = 4 * x / np.sqrt(n * np.sum(np.square(x)))
        part_1_const = np.exp(-0.2 * np.sqrt(1/n * np.sum(np.square(x))))
        part_2 = (2*np.pi/n * np.sin(2*np.pi*x))
        part_2_const = np.exp((1/n) * np.sum(np.cos(2*np.pi*x)))
        return part_1 * part_1_const + part_2 * part_2_const
        
    return ackley_grad

def rastrigin_gen(n):
    def rastrigin(x,*args):
        return 10 * n + np.sum(np.square(x) - 10 * np.cos(2*np.pi*x), axis=0)
    return rastrigin

def rast_deriv_gen(n):
    def rast_grad(x,*args):
        return 2 * x + 20 * np.pi * np.sin(2*np.pi*x)        
    return rast_grad
    
def himmelblau(x, *args):
    return np.square(np.square(x[0]) + x[1] - 11) + np.square(x[0] + np.square(x[1]) - 7)
    
def himmelblau_grad(x,*args):
    one = 2*2*x[0]*(x[0]**2+x[1]-11) + 2*(x[0]+x[1]**2-7)
    two = 2 * (x[0]**2+x[1]-11) + 2*2*x[1]*(x[0]+x[1]**2-7)
    return np.array([one,two])
    
def poly(x):
    return np.sum(np.square(x))
    
def poly_1(x):
    return np.square(x[0]) * x[1] + 2 * np.square(x[0]) * x[1]**2
    
def poly_1_grad(x):
    m,n = np.shape(x)
    grad_array = np.zeros((m,n))
    for i in range(m):
        x0 = x[i,:][0]
        x1 = x[i,:][1]
        grad_array[i,:] = np.array([2*x0*x1 + 4*x0*x1**2, x0**2 + 4*x0**2*x1])   
        
    return grad_array

def rosenbrock_gen(n, a=1, b=100, uncoupled=False):
    if uncoupled == True:
        if n%2 != 0:
            raise ValueError("Dimension for the uncoupled Rosenbrock function must be even")
        def rosenbrock(x, *args):
            x0 = x[::2]
            x1 = x[1::2]
            return np.sum(b*np.square(x1-np.square(x0)) + np.square(a*np.ones(int(n/2))-x0))
    else:
        def rosenbrock(x, *args):
            x0 = x[:-1]
            x1 = x[1:]
            return np.sum(b*np.square(x1-np.square(x0)) + np.square(a*np.ones(n-1)-x0))
    return rosenbrock

def rosen_deriv_gen(n, a=1, b=100, uncoupled=False):
    if uncoupled == True:
        if n%2 != 0:
            raise ValueError("Dimension for the uncoupled Rosenbrock function must be even")
        def rosen_grad(x, *args):
            grad = np.zeros(n)
            x0 = x[::2]
            x1 = x[1::2]
            
            grad[::2] = -4*b*(x1 - np.square(x0))*x0 - 2*(a*np.ones(int(n/2))-x0)
            grad[1::2] = 2*b*(x1 - np.square(x0))
            return grad
    else:
        def rosen_grad(x, *args):
            grad = np.zeros(n)
            grad2 = np.zeros(n)
            x0 = x[:-1]
            x1 = x[1:]

            grad[:-1] = -4*b*(x1 - np.square(x0))*x0 - 2*(a*np.ones(n-1)-x0)
            grad2[1:] = 2*b*(x1 - np.square(x0))
            grad += grad2
            return grad
    return rosen_grad
        
def zakharov_gen(n):
    def zakharov(x, *args):
        i = np.arange(1, n+1)
        isum = np.sum(0.5 * i * x)
        return np.sum(np.square(x)) + isum**2 + isum**4
    return zakharov
        
def zakharov_deriv_gen(n):
    def zakharov_grad(x, *args):
        i = np.arange(1, n+1)
        isum = np.sum(0.5 * i * x)
        coeff = 2*isum + 4*isum**3
        return 2*x + coeff * 0.5 * i
    return zakharov_grad
            
