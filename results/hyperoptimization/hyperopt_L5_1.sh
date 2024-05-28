#!/bin/bash

#SBATCH --time=10:00:00   # walltime
#SBATCH --ntasks=50   # number of processor cores (i.e. tasks)
#SBATCH --nodes=5  # number of nodes
#SBATCH --mem-per-cpu=5000M   # memory per CPU core
#SBATCH -J  "L5Test1"   # job name

module load miniconda3/4.6
source activate HypOpt

ray start --head
sleep 10
ray status


RAY_ADDRESS=auto python -u HypOpConstantLR.py --dim 2 --max_delay 5 --num_delays 10000 --use_delays True --maxiter 5000 \
--cost_function "ackley" --tol 1e-5 --num_runs 20 \
--filename "Ackley_Delayed_2_5" --num_initials 20 \
--num_processes 49 --bayesian_samples 5 --grid_samples 15 \
--ranges_0 0.0 --ranges_1 2.0

RAY_ADDRESS=auto python -u HypOpConstantLR.py --dim 2 --max_delay 5 --num_delays 10000 --use_delays True --maxiter 5000 \
--cost_function "rastrigin" --tol 1e-5 --num_runs 20 \
--filename "Rastrigin_Delayed_2_5" --num_initials 20 \
--num_processes 49 --bayesian_samples 5 --grid_samples 15 \
--ranges_0 0.0 --ranges_1 2.0

RAY_ADDRESS=auto python -u HypOpConstantLR.py --dim 2 --max_delay 5 --num_delays 10000 --use_delays True --maxiter 5000 \
--cost_function "zakharov" --tol 1e-5 --num_runs 20 \
--filename "Zakharov_Delayed_2_5" --num_initials 20 \
--num_processes 49 --bayesian_samples 5 --grid_samples 25 \
--ranges_0 0.0 --ranges_1 6.0

RAY_ADDRESS=auto python -u HypOpConstantLR.py --dim 2 --max_delay 5 --num_delays 10000 --use_delays True --maxiter 5000 \
--cost_function "rosenbrock" --tol 1e-5 --num_runs 20 \
--filename "Rosenbrock_Delayed_2_5" --num_initials 20 \
--num_processes 49 --bayesian_samples 5 --grid_samples 20 \
--ranges_0 0.0 --ranges_1 4.0




RAY_ADDRESS=auto python -u HypOpConstantLR.py --dim 10 --max_delay 5 --num_delays 10000 --use_delays True --maxiter 5000 \
--cost_function "ackley" --tol 1e-5 --num_runs 20 \
--filename "Ackley_Delayed_10_5" --num_initials 20 \
--num_processes 49 --bayesian_samples 5 --grid_samples 15 \
--ranges_0 0.0 --ranges_1 2.0

RAY_ADDRESS=auto python -u HypOpConstantLR.py --dim 10 --max_delay 5 --num_delays 10000 --use_delays True --maxiter 5000 \
--cost_function "rastrigin" --tol 1e-5 --num_runs 20 \
--filename "Rastrigin_Delayed_10_5" --num_initials 20 \
--num_processes 49 --bayesian_samples 5 --grid_samples 15 \
--ranges_0 0.0 --ranges_1 2.0

RAY_ADDRESS=auto python -u HypOpConstantLR.py --dim 10 --max_delay 5 --num_delays 10000 --use_delays True --maxiter 5000 \
--cost_function "zakharov" --tol 1e-5 --num_runs 20 \
--filename "Zakharov_Delayed_10_5" --num_initials 20 \
--num_processes 49 --bayesian_samples 5 --grid_samples 25 \
--ranges_0 0.0 --ranges_1 6.0

RAY_ADDRESS=auto python -u HypOpConstantLR.py --dim 10 --max_delay 5 --num_delays 10000 --use_delays True --maxiter 5000 \
--cost_function "rosenbrock" --tol 1e-5 --num_runs 20 \
--filename "Rosenbrock_Delayed_10_5" --num_initials 20 \
--num_processes 49 --bayesian_samples 5 --grid_samples 20 \
--ranges_0 0.0 --ranges_1 4.0




RAY_ADDRESS=auto python -u HypOpConstantLR.py --dim 100 --max_delay 5 --num_delays 10000 --use_delays True --maxiter 5000 \
--cost_function "ackley" --tol 1e-5 --num_runs 20 \
--filename "Ackley_Delayed_100_5" --num_initials 20 \
--num_processes 49 --bayesian_samples 5 --grid_samples 15 \
--ranges_0 0.0 --ranges_1 2.0

RAY_ADDRESS=auto python -u HypOpConstantLR.py --dim 100 --max_delay 5 --num_delays 10000 --use_delays True --maxiter 5000 \
--cost_function "rastrigin" --tol 1e-5 --num_runs 20 \
--filename "Rastrigin_Delayed_100_5" --num_initials 20 \
--num_processes 49 --bayesian_samples 5 --grid_samples 15 \
--ranges_0 0.0 --ranges_1 2.0

RAY_ADDRESS=auto python -u HypOpConstantLR.py --dim 100 --max_delay 5 --num_delays 10000 --use_delays True --maxiter 5000 \
--cost_function "zakharov" --tol 1e-5 --num_runs 20 \
--filename "Zakharov_Delayed_100_5" --num_initials 20 \
--num_processes 49 --bayesian_samples 5 --grid_samples 25 \
--ranges_0 0.0 --ranges_1 6.0

RAY_ADDRESS=auto python -u HypOpConstantLR.py --dim 100 --max_delay 5 --num_delays 10000 --use_delays True --maxiter 5000 \
--cost_function "rosenbrock" --tol 1e-5 --num_runs 20 \
--filename "Rosenbrock_Delayed_100_5" --num_initials 20 \
--num_processes 49 --bayesian_samples 5 --grid_samples 20 \
--ranges_0 0.0 --ranges_1 4.0

ray stop
python add_hyperparam_data.py

conda deactivate
