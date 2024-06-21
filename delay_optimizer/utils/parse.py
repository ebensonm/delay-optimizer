from ..optimization import optimizers, schedulers, functions
from ..delays import distributions


SCHED_KEYS = {"lr", "max_lr", "min_lr", "gamma", "step_size", "p"}
OPT_KEYS = {"momentum", "beta_1", "beta_2", "epsilon"}
DELAY_KEYS = {"max_L", "num_delays", "D"}

def parse_kwargs(kwargs):
    scheduler_kwargs = {}
    optimizer_kwargs = {}
    delay_kwargs = {}

    for k, v in kwargs.items():
        # Special cases
        if k == "learning_rate":
            k = "lr"
        elif k in ("beta1", "beta-1"):
            k = "beta_1"
        elif k in ("beta2", "beta-2"):
            k = "beta_2"

        elif k in SCHED_KEYS:
            scheduler_kwargs[k] = v
        elif k in OPT_KEYS:
            optimizer_kwargs[k] = v
        elif k in DELAY_KEYS:
            delay_kwargs[k] = v
        else:
            raise ValueError(f"Could not parse key: {k}")
    return objective_kwargs, scheduler_kwargs, optimizer_kwargs, delay_kwargs

def parse_objective_function(self, objective, **kwargs):
    match objective:
        case functions.ObjectiveFunction:
            return objective
        case str():
            objective = objective.lower()
            if objective == "ackley":
                return functions.Ackley(**kwargs)
            elif objective == "rastrigin":
                return functions.Rastrigin(**kwargs)
            elif objective == "rosenbrock":
                return functions.Rosenbrock(**kwargs)
            elif objective == "zakharov":
                return functions.Zakharov(**kwargs)
            else:
                raise ValueError("String input does not match any known objective function.")
        case _:
            raise ValueError("Invalid input for objective function.")

def parse_scheduler(self, scheduler, **kwargs):
    match scheduler:
        case schedulers.Scheduler:
            return scheduler
        case str():
            scheduler = scheduler.lower()
            if scheduler == "constant":
                return schedulers.Constant(**kwargs)
            elif scheduler == "step":
                return schedulers.Step(**kwargs)
            elif scheduler == "inv":
                return schedulers.Inv(**kwargs)
            elif scheduler in ("tri", "tri2", "tri-2", "tri_2"):
                return schedulers.Tri2(**kwargs)
            elif scheduler in ("sin", "sin2", "sin-2", "sin_2"):
                return schedulers.Sin2(**kwargs)
            else:
                raise ValueError("String input does not match any known learning rate scheduler.")
        case _:
            raise ValueError("Invalid input for learning rate scheduler.")

def parse_optimizer(self, optimizer, **kwargs):
    match optimizer:
        case optimizers.Optimizer:
            return optimizer
        case str():
            optimizer = optimizer.lower()
            if optimizer == "adam":
                return optimizers.Adam(**kwargs)
            elif optimizer == "momentum":
                return optimizers.Momentum(**kwargs)
            elif optimizer == "gradientdescent":
                return optimizers.GradientDescent(**kwargs)
            else:
                raise ValueError("String input does not match any known optimizer.")
        case _:
            raise ValueError("Invalid input for optimizer.")

def parse_delay_distribution(self, delays, **kwargs):
    match delays:
        case distributions.DelayType:
            return delays
        case str():
            delays = delays.lower()
            if delays == "undelayed":
                return distributions.Undelayed(**kwargs)
            elif delays == "uniform":
                return distributions.Uniform(**kwargs)
            elif delays == "stochastic":
                return distributions.Stochastic(**kwargs)
            elif delays == "decaying":
                return distributions.Decaying(**kwargs)
            elif delays == "partial":
                return distributions.Partial(**kwargs)
            elif delays == "cyclical":
                return distributions.Cyclical(**kwargs)
            elif delays == "constant":
                return distributions.Constant(**kwargs)
            else:
                raise ValueError("String input does not match any known delay distribution.")
        case _:
            raise ValueError("Invalid input for delay distribution.")