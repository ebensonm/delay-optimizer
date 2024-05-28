from computation import Computation
from Optimizer_Scripts.DelayTypeGenerators import *

# Optimal Learning Rates --------------------------------------------------

# Ackley
ack_undel_lrs = {2: 1.85721428571429,
                 10: 0.356360766680998,
                 100: 0.143927494378293,
                 1000: 0.087579721240169}
ack_stochL1_lrs = {2: 1.33067690501033,
                   10: 1.42885714285714,
                   100: 1.44148801018898,
                   1000: 1.40918128594683}

# Rastrigin
rast_undel_lrs = {2: 0.510334818569094,
                  10: 0.513701966615492,
                  100: 0.51114689564692,
                  1000: 0.518119401111126}
rast_stochL1_lrs = {2: 0.506103165038649,
                    10: 0.578173243431346,
                    100: 0.572142857142857,
                    1000: 0.572142857142857}

# Rosenbrock
rosen_undel_lrs = {2: 3.94647350971932,
                   10: 0.566837845876878,
                   100: 0.329639308828399,
                   1000: 0.310612039345774}
rosen_stochL1_lrs = {2: 3.57905263157895,
                     10: 0.421947368421053,
                     100: 0.266178324489324,
                     1000: 0.268033972561053}

# Zakharov
zak_undel_lrs = {2: 1.73889481723777,           # For maxiter = 10000
                 10: 3.97646574749106,          #     num_points = 1000
                 100: 2.41455172413793, 
                 1000: 0.986621657063249}
zak_stochL1_lrs = {2: 4.125375,                 # For maxiter = 10000
                   10: 2.76109419287542,        #     num_points = 1000
                   100: 0.391382192771158,
                   1000: 5.17289655172414}


# Define computation objects ----------------------------------------------

ack = Computation("Ackley", num_points=250)
rast = Computation("Rastrigin", num_points=250)
rosen = Computation("Rosenbrock", num_points=250)
zak = Computation("Zakharov", num_points=1000)


"""  Syntax for computation
comp.run_save(d, delay_type, lr_type='const', file_tag="", overwrite=False, **kwargs)
comp.run_save_all(delay_type, file_tag="", overwrite=False, lrs=None, dimensions=[2,10,100,1000], **kwargs)
"""

# DONT FORGET TO SET MAXITER=10000 FOR ZAKHAROV

class Custom(DelayType):
    def __init__(self, max_L, num_delays, del_size=900, undel_size=100, **kwargs):
        DelayType.__init__(self, max_L, num_delays)
        self.delay_type = "custom"
        self.del_size = del_size
        self.undel_size = undel_size
    
    def D_gen(self, n):
        i = 0 
        while i < self.num_delays:
            for j in range(self.del_size):
                yield np.random.randint(0, self.max_L+1, n)
                i += 1
                if i >= self.num_delays:
                    break
            for k in range(self.undel_size):
                yield np.zeros(n, dtype=int)
                i += 1
                if i >= self.num_delays:
                    break
        while True:
            yield np.zeros(n, dtype=int)
            

# Test on the Ackley function
from Analyzer import FuncOpt

loss_name = "Zakharov"
dim = 1000
custom_del = Custom(1, 5000, 900, 100)
ack_opt = FuncOpt(loss_name, dim)
ack_opt.random_points(1)
ack_opt.optimize(custom_del, "const", maxiter=5000, 
                 learning_rate=zak_stochL1_lrs[1000])  # NOTE: Change this lr
ack_opt.save_data(r"Data/{}{}d_stochL1_custom_test".format(loss_name, dim))








