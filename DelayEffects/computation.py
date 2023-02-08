# computation.py

import sys
#sys.path.append('/home/yungdankblast/Desktop/Research/delay-optimizer')
sys.path.append('/home/cayjobla/Desktop/Research/delay-optimizer')
from Analyzer import FuncOpt
from Optimizer_Scripts.DelayTypeGenerators import *
from Optimizer_Scripts.Data import Data
import os

class Computation:
    
    def __init__(self, loss_name, dimensions=[2,10,100,1000]):
        self.loss_name = loss_name
        self.dimensions = dimensions
        self.func_dict = {d:FuncOpt(loss_name, d) for d in self.dimensions}
        

    def gen_points(self, num_points):
        for d in self.dimensions:
            self.func_dict[d].random_points(num_points)
            if d > 2:
                self.func_dict[d].x_inits[:,:2] = self.func_dict[2].x_inits
            
    def load_points(self, filename):
        data = Data.load(filename)
        x_inits = data.get_x_inits()
        self.func_dict[2].load_points(x_inits) 
        
        for d in self.dimensions:
            if d > 2:
                self.func_dict[d].random_points(len(x_inits))
                self.func_dict[d].x_inits[:,:2] = x_inits
    
    
    def run_save(self, d, delay_type, lr_type='const', file_tag="", 
                 overwrite=False, **kwargs):
        func_obj = self.func_dict[d]
        filename = r"Data/{}{}d_{}.dat".format(self.loss_name, d, file_tag)
        
        if overwrite == False and os.path.isfile(filename):
            print(filename, "already exists. Use overwrite=True if you wish "
                  "to overwrite the previous data.")
            return
        
        func_obj.optimize(delay_type, lr_type, **kwargs)    # Optimize
        func_obj.save_data(filename)                        # Save values
        func_obj.delete_data() 
    
            
    def run_save_all(self, delay_type, file_tag="", overwrite=False, lrs=None, 
                     **kwargs):
        for i, d in enumerate(self.dimensions):
            if lrs is not None:
                self.run_save(d, delay_type.copy(), file_tag, overwrite, 
                              learning_rate=lrs[i], **kwargs)
            else:
                self.run_save(d, delay_type.copy(), file_tag, overwrite, 
                              learning_rate=lrs[i], **kwargs)
            
                  
    
    
    
    
    
    
    