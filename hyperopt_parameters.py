from Optimizer_Scripts.functions import ackley_gen, rastrigin_gen, ackley_deriv_gen, rast_deriv_gen, poly, poly_1, poly_1_grad
from Optimizer_Scripts.functions import zakharov_gen, zakharov_deriv_gen, rosenbrock_gen, rosen_deriv_gen
import argparse

def str2bool(v):
    """Convert the input strings to boolean
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def get_arguments(raw_args=None):

    #get the argument parser
    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dim',
                        '-n',
                        help=("The test dimension"),
                        type=int,
                        default=2
    )
    parser.add_argument('--max_delay',
                        '-l',
                        help=("The maximum delay"),
                        type=int,
                        default=1
    )
    parser.add_argument('--num_delays',
                        '-d',
                        help=("The number of total delays to have before setting all to 0"),
                        type=int,
                        default=1000
    )
    parser.add_argument('--use_delays', 
                         '-u',
                         nargs='?',
                         help=("Use delays in the optimizer"),
                         type=str2bool,
                         default=True
    )
    parser.add_argument('--maxiter',
                        '-m',
                        help=("The maximum number of iterations to run on optimizer calling a loss"),
                        type=int,
                        default=5000
    )
    parser.add_argument('--cost_function',
                        '-c',
                        help=("The loss or cost function to optimize over"),
                        type=str,
                        default="ackley"
    )
    parser.add_argument('--tol',
                        '-t',
                        help=("The tolerance for convergence in the optimizer"),
                        type=float,
                        default=1e-5
    )
    parser.add_argument('--num_runs',
                        '-e',
                        help=("The number of bayesian hyperparameter evaluations to do"),
                        type=int,
                        default=10
    )
    parser.add_argument('--filename',
                        '-f',
                        help=("Name of file to store results"),
                        type=str, 
                        default='2d_test',
    )
    parser.add_argument('--num_initials',
                        '-y',
                        help=("Number of total initial values to test per cpu"),
                        type=int, 
                        default=20
    )
    parser.add_argument('--num_processes',
                        '-g',
                        help=("Total number of available cpu's"),
                        type=int, 
                        default=7
    )
    parser.add_argument('--bayesian_samples',
                        '-b',
                        help=("Number of initital sample points for Bayesian optimization"),
                        type=int, 
                        default=5
    )
    parser.add_argument('--grid_samples',
                        '-w',
                        help=("Number of initital sample points for grid search optimization"),
                        type=int, 
                        default=5
    )
    parser.add_argument('--max_range_0',
                        '-dl',
                        help=("The lowest value of the min learning rate sampling range"),
                        type=float, 
                        default=None
    )
    parser.add_argument('--max_range_1',
                        '-bc',
                        help=("The highest value of the max learning rate sampling range"),
                        type=float,
                        default=None
    )
    parser.add_argument('--min_range_0',
                        '-hd',
                        help=("The lowest value of the min learning rate sampling range"),
                        type=float, 
                        default=None
    )
    parser.add_argument('--min_range_1',
                        '-i',
                        help=("The highest value of the min learning rate sampling range"),
                        type=float, 
                        default=None
    )
    parser.add_argument('--lr_type',
                        '-q',
                        help=("The type of learning rate schedule to use"),
                        type=str, 
                        default='step'
    )
    parser.add_argument('--gamma_1',
                        '-x',
                        help=("the lowest value of the gamma sampling range"),
                        type=float,
                        default=None
    )
    parser.add_argument('--gamma_2',
                        '-v',
                        help=("the highest value of the gamma sampling range"),
                        type=float,
                        default=None
    )
    parser.add_argument('--p_1',
                        '-o',
                        help=("The lowest value of the p sampling range"),
                        type=float,
                        default=None
    )
    parser.add_argument('--p_2',
                        '-j',
                        help=("The highest value of the p sampling range"),
                        type=float,
                        default=None
    )
    parser.add_argument('--step_1',
                        '-s',
                        help=("The lowest value of the step size sampling range"),
                        type=float,
                        default=None
    )
    parser.add_argument('--step_2',
                        '-tz',
                        help=("The highest value of the step size sampling range"),
                        type=float,
                        default=None
    )    
    args = parser.parse_args(raw_args)
    return args

def get_cost_function(function_name='ackley', dimension=10):
    
     if function_name == 'ackley':
         domain_0 = -32.0
         domain_1 = 32.0
         cost_function = ackley_gen(dimension)
         gradient = ackley_deriv_gen(dimension)
     elif function_name == 'rastrigin':
         domain_0 = -5.12
         domain_1 = 5.12
         cost_function = rastrigin_gen(dimension)
         gradient = rast_deriv_gen(dimension)
     elif function_name == 'zakharov':
         domain_0 = -5.0
         domain_1 = 10.0
         cost_function = zakharov_gen(dimension)
         gradient = zakharov_deriv_gen(dimension)
     elif function_name == 'rosenbrock':
         domain_0 = -5.0
         domain_1 = 10.0
         cost_function = rosenbrock_gen(dimension)
         gradient = rosen_deriv_gen(dimension)
     else:
         raise ValueError("loss function name must be either 'ackley', 'rastrigin', zakharov', or 'rosenbrock'")
     
     return cost_function, gradient, domain_0, domain_1
     
