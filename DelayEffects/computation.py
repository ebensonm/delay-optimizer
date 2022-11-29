# computation.py

import sys
#sys.path.append('/home/yungdankblast/Desktop/Research/delay-optimizer')
sys.path.append('/home/cayjobla/Desktop/Research/delay-optimizer')
from Analyzer_new import FuncOpt
from Optimizer_Scripts.DelayTypeGenerators import Undelayed, Stochastic
import os

class Computation:
    
    def __init__(self, loss_name, dimensions=[2,10,100,1000], **kwargs):
        self.loss_name = loss_name
        self.dimensions = dimensions
        self.func_dict = {d:FuncOpt(loss_name, d, **kwargs) for d in self.dimensions}
        

    def gen_points(self, num_points):
        for d in self.dimensions:
            self.func_dict[d].initialize_points('random', num_points)
            if d > 2:
                self.func_dict[d].x_inits[:,:2] = self.func_dict[2].x_inits
            
    
    def run_save(self, d, delay_type, param_type='optimal', write='new', 
                 save_dims=(0,1), file_tag="", **kwargs):
        func_obj = self.func_dict[d]
        filename = r"Data/{}{}d_{}.npz".format(self.loss_name, d, file_tag)
        
        if write != 'overwrite' and os.path.isfile(filename):
            print(filename, "already exists. Use write='overwrite' if you "
                  "wish to overwrite the previous data.")
            return
        
        func_obj.optimize(delay_type, param_type, **kwargs)       # Perform optimization 
        func_obj.save(filename, save_dims)                        # Save values
        func_obj.delete_data() 
    
            
    def run_save_all(self, param_dict):
        for d, params in param_dict.items():
            self.run_save(d, **params)

def test():
    func = Computation("Zakharov")
    func.gen_points(250)
    params = {'param_type':'given', 
              'write':'new', 
              'file_tag':"stoch_const_1.0", 
              "learning_rate": 1.}
    full_run_params = {d:params.copy() for d in func.dimensions}
    for d in func.dimensions:
        full_run_params[d]['delay_type'] = Stochastic(1,2000)
    func.run_save_all(full_run_params)
                  
    
    
    
    
    
    
    