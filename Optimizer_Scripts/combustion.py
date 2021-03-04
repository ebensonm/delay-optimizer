import julia
from julia import Main as Combustion
import numpy as np 
   
def functional_wrapper(function):
    def f(x):
        x = x.tolist()
        #handle out of range derivative computation with try except block
        try:
           value = np.asarray(function(x))
        except:
           return None
        return value
    return f

def get_combustion_model(vary_percent=0.1, path="Optimizer_Scripts/Combustion_Model/Model.jl", n=216):
    np.random.seed(12)
    #load the model
    print("Loading Combustion Model")
    Combustion.include(path) 
    #get the minimizer
    x_min = np.array(Combustion.x)
    #set initial value
    mult_array = 1.0 + np.random.uniform(-vary_percent,vary_percent,size=n)
    x_init = x_min * mult_array
    #get the objective function and the gradient
    objective = functional_wrapper(Combustion.objective)
    gradient = functional_wrapper(Combustion.gradient)
    #return usefull values
    print("Returning Combustion functions")
    return x_min, x_init, objective, gradient
