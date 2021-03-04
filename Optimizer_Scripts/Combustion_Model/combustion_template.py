import julia
from julia import Main as Combustion
import numpy as np


if __name__=="__main__":
    print("Getting File")
    Combustion.include("Model.jl")   
     
    print("Done!")
       
    try:
        print(Combustion.objective)
    except:
        print("Test 2: Failed!")
       
    try:
        print(Combustion.gradient)
    except:
        print("Test 3: Failed!")
       
    try:
        print(len(Combustion.x))
    except:
        print("Test 4: Failed!")
       
    try:
        print(Combustion.gradient(Combustion.x))
    except:
        print("Test 5: Failed!")

    try:
        x_init = np.random.uniform(size=(3,216)).tolist()
        for init in x_init:
            print(Combustion.objective(init))
    except:
        print("Test 6: Failed!")         
         
    try:
        x_init = np.random.uniform(size=(3,216)).tolist()
        for init in x_init:
            print(Combustion.gradient(init))
    except:
        print("Test 7: Failed!")
       
    try:
        print(Combustion.objective(Combustion.x))
    except:
        print("Test 8: Failed!")
