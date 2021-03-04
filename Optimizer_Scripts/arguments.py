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
    parser.add_argument('--max_L',
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
    parser.add_argument('--optimizer_name',
                        '-o',
                        help=("The optimizer to use in the test"),
                        type=str,
                        default="Adam"
    )
    parser.add_argument('--loss_name',
                        '-c',
                        help=("The loss or cost function to optimize over"),
                        type=str,
                        default="Rastrigin"
    )
    parser.add_argument('--tol',
                        '-t',
                        help=("The tolerance for convergence in the optimizer"),
                        type=float,
                        default=1e-5
    )
    parser.add_argument('--max_evals',
                        '-e',
                        help=("The number of hyperparameter evaluations to do"),
                        type=int,
                        default=300
    )
    parser.add_argument('--symmetric_delays',
                        '-s',
                        help=("Use symmetric delays, to maximize computation time"),
                        type=str2bool,
                        default=True
    )
    parser.add_argument('--constant_learning_rate',
                        '-r',
                        help=("Constant or nonconstant learning rate"),
                        type=str2bool, 
                        default=False
    )
    parser.add_argument('--num_tests',
                        '-z',
                        help=("Number of tests to run"),
                        type=int, 
                        default=1
    )
    parser.add_argument('--filename',
                        '-f',
                        help=("Name of file to store results"),
                        type=str, 
                        default='2d_test',
    )
    parser.add_argument('--num_test_initials',
                        '-y',
                        help=("Number of total initial values to test per cpu"),
                        type=int, 
                        default=20
    )
    parser.add_argument('--logging',
                        '-g',
                        help=("Whether or not optimizer will print steps in the optimization process"),
                        type=str2bool,
                        default=False
    )
    parser.add_argument('--vary_percent',
                        '-v',
                        help=("Argument for varying initil value from minimizer in combustion problem"),
                        type=float,
                        default=0.1
    )
    parser.add_argument('--hyper_minimize',
                        '-x',
                        help=("Which value to minimize with hyperparameter optimization"),
                        choices=["loss", "distance"],
                        default="loss"
    )
    
    args = parser.parse_args(raw_args)
    return args
