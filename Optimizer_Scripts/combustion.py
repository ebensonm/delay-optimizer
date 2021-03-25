#from julia.api import Julia
#jl = Julia(compiled_modules=False)
from julia import Main
import numpy as np 
   
def functional_wrapper(function, as_array=True):
    def f(x):
        x = x.tolist()
        #handle out of range derivative computation with try except block
        try:
           value = function(x)
           if (as_array is True):
               value = np.asarray(value)
        except:
           return None
        return value
    return f

def get_combustion_model(vary_percent=0.1, path="Optimizer_Scripts/Combustion_Model/Model.jl", n=216):
    np.random.seed(12)
    #load the model
    print("Loading Combustion Model")
    Main.include(path)
    CombustionModel = Main.CombustionModel 
    #get the minimizer
    x_min = np.array(CombustionModel.x)
    #set initial value
    mult_array = 1.0 + np.random.uniform(-vary_percent,vary_percent,size=n)
    x_init = x_min * mult_array
    #get the objective function and the gradient
    objective = functional_wrapper(CombustionModel.objective, as_array=False)
    gradient = functional_wrapper(CombustionModel.gradient)
    #return useful values
    print("Returning Combustion functions")
    return x_min, x_init, objective, gradient
