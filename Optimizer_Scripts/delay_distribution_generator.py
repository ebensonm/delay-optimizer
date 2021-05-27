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
            
            
def get_distribution(constant=True, D=None):
    """
    The handler for using the functions in this script for generating different types of nonstochastic
    delay distributions
    """
    if (constant == True):
        return constant_distribution(D=D)
    return cyclical_distribution(D=D)
