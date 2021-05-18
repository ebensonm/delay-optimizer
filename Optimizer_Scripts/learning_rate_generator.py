def const_lr_gen(params):
    GO = True
    learning_rate = params['learning_rate']
    while GO is True:
       yield learning_rate

def non_const_lr_gen(params):  
    #dummy variable for running the generator
    GO = True
    #counter for the iterations
    counter = 0
    #variable for how many iterations per triangle side
    lr_num = params['step_size']
    min_lr = params['min_learning_rate']
    max_lr = params['max_learning_rate']
    #compute the step size
    dh = (max_lr - min_lr)/lr_num
    #the learning rate starter
    learning_rate = params['min_learning_rate'] - dh
    counter_counter = 0
    while GO is True:
        counter += 1
        if (counter % lr_num == 0):
            counter_counter += 1
            #shrinking triangle part of the changing learning rate
            if (counter_counter % 2 == 0):
                max_lr -= (max_lr-min_lr)/2
                dh = (max_lr - min_lr)/lr_num
                counter_counter = 0
            else:
                dh = dh*-1
            counter = 0
        learning_rate += dh
        yield learning_rate      
        
def generate_learning_rates(constant_lr, params):
    """ Create the learning rate generator for constant and nonconstant learning rates
    """
    if (constant_lr is True):
        return const_lr_gen(params)
    else:
        return non_const_lr_gen(params)
