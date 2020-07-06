import numpy as np
from Optimizer_Scripts.functions import ackley_gen, rastrigin_gen, ackley_deriv_gen, rast_deriv_gen
from Optimizer_Scripts.optimizers import Adam
from Optimizer_Scripts.Delayer import Delayer

#things we can try to reduce computational complexity
# - equivalent rows of delay matrix, so we only have to do all the computations once
# - does the ackley function even work?              
                           
if __name__ == "__main__":
    n = 2                             #the number of dimensions of the state vector
    max_L = 1                         #the max delay that the system can have
    num_delays = 8500                 #the number of delays before removing all delays
    max_iter = 10000
    x_init = np.random.uniform(-5.12,5.12,n)  #starting state values
    adam_optimizer = Adam()           #optimizer to add time delays to
    loss_function = rastrigin_gen(n)  #loss function
    deriv_loss = rast_deriv_gen(n)    #derivative of loss function
    #initialize the class with a call to Delayer, which adds the delays
    delayer = Delayer(n, adam_optimizer, loss_function, deriv_loss, x_init, max_L, num_delays)  
    delayer.compute_time_series(use_delays=True,maxiter=max_iter) #computes the states over time 
    print(np.linalg.norm(delayer.final_state))  #print the final state vector
    print(delayer.final_val)                    #print the final loss
    print(delayer.conv)                         #print whether or not it converged
    print(len(delayer.time_series))             #print the number of iterations before convergence
