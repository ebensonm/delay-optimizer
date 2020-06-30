import numpy as np
from Optimizer_Scripts.functions import ackley_gen, rastrigin_gen, ackley_deriv_gen, rast_deriv_gen
from Optimizer_Scripts.optimizers import Adam
from Optimizer_Scripts.Delayer import Delayer

#things we can try to reduce computational complexity
# - equivalent rows of delay matrix, so we only have to do all the computations once
# - does the ackley function even work?              
                           
if __name__ == "__main__":
    n = 100
    max_L = 1
    num_delays = 1200
    x_init = np.random.uniform(-5.12,5.12,n)
    #x_init = np.random.uniform(-32.,32.,n)
    adam_optimizer = Adam()
    #loss function to add
    loss_function = rastrigin_gen(n)
    #loss_function = ackley_gen(n)
    #derivative of loss function
    deriv_loss = rast_deriv_gen(n)
    #deriv_loss = ackley_deriv_gen(n)
    delayer = Delayer(n, adam_optimizer, loss_function, deriv_loss, x_init, max_L, num_delays)
    delayer.compute_time_series(use_delays=True)
    print(np.linalg.norm(delayer.final_state))
    print(delayer.final_val)
    print(delayer.conv)
    print(len(delayer.time_series))
