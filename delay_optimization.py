import numpy as np
from Optimizer_Scripts.functions import ackley_gen, rastrigin_gen, ackley_deriv_gen, rast_deriv_gen
from Optimizer_Scripts.optimizers import Adam, Momentum, NesterovMomentum
from Optimizer_Scripts.Delayer import Delayer
from Optimizer_Scripts.hyper_param_optimizers import adam_optimizer_optimizer, momentum_optimizer_optimizer, nesterov_optimizer_optimizer
from hyperopt import hp, tpe, fmin, Trials
import multiprocessing as mp
import dill

#things we can try to reduce computational complexity
# - equivalent rows of delay matrix, so we only have to do all the computations once
# - does the ackley function even work?  

def use_rast(n,max_L,num_delays,optimizer):
    np.random.seed(12)
    x_init = np.random.uniform(-5.12,5.12,n)
    loss_function = rastrigin_gen(n)
    deriv_loss = rast_deriv_gen(n)
    delayer = Delayer(n, optimizer, loss_function, deriv_loss, x_init, max_L, num_delays)
    return delayer
    
def use_ackley(n,max_L,num_delays,optimizer):
    np.random.seed(12)            
    x_init = np.random.uniform(-32.,32.,n)
    loss_function = ackley_gen(n)
    deriv_loss = ackley_deriv_gen(n)
    delayer = Delayer(n, optimizer, loss_function, deriv_loss, x_init, max_L, num_delays)
    return delayer
                      
if __name__ == "__main__":
    n = 100
    #max_L = 10
    num_delays = 800
    use_delays = True
    optimizer = Adam()
    object_list = list()
    for i in range(1,11):
        max_L = i * 10
        delayer = use_ackley(n,max_L,num_delays,optimizer)
        object_list.append(delayer)
         
    def multi_test(i,delayer):
        alpha, beta_1, beta_2 = momentum_optimizer_optimizer(delayer, use_delays=use_delays)
        delayer.Optimizer.learning_rate = alpha
        delayer.Optimizer.beta_1 = beta_1
        delayer.Optimizer.beta_2 = beta_2
        delayer.delete_time_series()
        with open('../results/tests_4/object_100_u_'+str(i+1)+'.pkl','wb') as inFile:
            dill.dump(delayer,inFile)      
        del delayer         
    processes = []
    for i in range(0,8):
        p = mp.Process(target=multi_test,args=(i,object_list[i],))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
