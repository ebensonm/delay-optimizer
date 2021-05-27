import numpy as np


def constant_distribution(D=None):
    """
    uses a given constant D (delay distribution) to create the generator yielding the
    same delay distribution everytime
    """
    go = True
    while (go == True):
        yield D


def cyclical_distribution(D=list()):
    """
    uses a list of delay distributions and does cyclical distribution generation looping through
    each item in the list before starting over again
    """
    go = True
    while (go == True):
        for dist in D:
            yield dist
            
            
def stochastic_distribution(n=2, max_L=1, symmetric=True, shrink=False):
   """
   the stochastic time delay distribution generator to be used in the time delayed
   optimizer class
   """
   # TODO - add shrinking capability to the stochastic delay distribution generator 
   #  (iterval = i+1)
   #if (iter_val % (self.num_delays // self.max_L) == 0 and shrink==True):    #shrink delays every interval
   #     self.num_max_delay -= 1
   
   go = True
   num_max_delay = max_L
   while (go == True):
       if (symmetric is True):
           D = np.random.randint(0, num_max_delay+1,n)
       else:
           D = np.random.randint(0,num_max_delay+1,n**2)
       yield D
            
            
def get_distribution(random=True, D=None, n=2, max_L=1, symmetric=True, shrink=False):
    """
    The handler for using the functions in this script for generating different types of nonstochastic
    delay distributions
    """
    if (random == True):
        return stochastic_distribution(n=n, max_L=max_L, symmetric=symmetric, shrink=shrink)
    elif (len(D) == 1):
        return constant_distribution(D=D)
    else:
        return cyclical_distribution(D=D)
