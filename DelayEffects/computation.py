# computation.py

import sys
#sys.path.append('/home/yungdankblast/Desktop/Research/delay-optimizer')
sys.path.append('/home/cayjobla/Desktop/Research/delay-optimizer')
import os
import numpy as np
from Analyzer import FuncOpt

from Optimizer_Scripts.DelayTypeGenerators import *


class Computation:
    
    def __init__(self, loss_name, num_points=250, load_points=True, **kwargs):
        self.loss_name = loss_name
        
        if load_points: 
            points_tag = "points_" + str(num_points)
            filename = self.get_filename(file_tag=points_tag)
            self.load_points(filename)
        else: 
            self.gen_points(num_points, **kwargs)
            
        
    def get_analyzer(self, d):
        func_opt = FuncOpt(self.loss_name, d)
        if hasattr(self, 'x_inits'):
            func_opt.load_points(self.x_inits[:,:d])
        return func_opt
    

    def gen_points(self, num_points, max_dim=1000):
        func_opt = self.get_analyzer(max_dim)
        func_opt.random_points(num_points)
        self.x_inits = func_opt.x_inits
        
            
    def load_points(self, filename):
        self.x_inits = np.load(filename, allow_pickle=True)['x_inits']
        
                
    def save_points(self, filename=None):
        if filename is None:
            points_tag = "points_" + str(len(self.x_inits))
            filename = self.get_filename(file_tag=points_tag)
        np.savez_compressed(filename, x_inits=self.x_inits)
        
        
    def get_filename(self, dim=None, file_tag="", ext=".npz", path="Data/"):
        if dim is None:
            return path + self.loss_name + "_" + file_tag + ext
        else:
            return path + self.loss_name + str(dim) + "d_" + file_tag + ext
        
    
    def run_save(self, d, delay_type, lr_type='const', file_tag="", 
                 overwrite=False, **kwargs):
        func_opt = self.get_analyzer(d)
        filename = self.get_filename(d, file_tag, ".dat")
        
        if overwrite == False and os.path.isfile(filename):
            print(filename, "already exists. Use overwrite=True if you wish "
                  "to overwrite the previous data.")
            return
        
        func_opt.optimize(delay_type, lr_type, **kwargs)    # Optimize
        func_opt.save_data(filename)                        # Save values
        func_opt.delete_data() 
        
            
    def run_save_all(self, delay_type, file_tag="", overwrite=False, lrs=None, 
                     dimensions=[2,10,100,1000], **kwargs):
        for i, d in enumerate(dimensions):
            if lrs is not None:
                self.run_save(d, delay_type.copy(), 'const', file_tag, 
                              overwrite, learning_rate=lrs[i], **kwargs)
            else:
                self.run_save(d, delay_type.copy(), 'const', file_tag, 
                              overwrite, **kwargs)
            
    
    
    
    
    