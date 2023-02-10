#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=50   # number of processor cores (i.e. tasks)
#SBATCH --nodes=5  # number of nodes
#SBATCH --mem-per-cpu=8000M   # memory per CPU core
#SBATCH -J  "ZTestU"   # job name

module load miniconda3/4.6
source activate HypOpt

ray start --head
sleep 10
ray status
RAY_ADDRESS=auto python -u HypOpConstantLR.py \
--dim 10 --max_delay 1 --num_delays 10000 \
--use_delays False --maxiter 10000 \
--cost_function "zakharov" --tol 1e-5 --num_runs 10 \
--filename "Zakharov_UnDelayed0_10" --num_initials 10 \
--num_processes 25 --bayesian_samples 5 --grid_samples 30 \
--max_range_0 0.0 --max_range_1 10.0

RAY_ADDRESS=auto python -u HypOpConstantLR.py \
--dim 100 --max_delay 1 --num_delays 10000 \
--use_delays False --maxiter 10000 \
--cost_function "zakharov" --tol 1e-5 --num_runs 10 \
--filename "Zakharov_UnDelayed0_100" --num_initials 10 \
--num_processes 25 --bayesian_samples 5 --grid_samples 30 \
--max_range_0 0.0 --max_range_1 10.0

RAY_ADDRESS=auto python -u HypOpConstantLR.py \
--dim 1000 --max_delay 1 --num_delays 10000 \
--use_delays False --maxiter 10000 \
--cost_function "zakharov" --tol 1e-5 --num_runs 10 \
--filename "Zakharov_UnDelayed0_1000" --num_initials 10 \
--num_processes 25 --bayesian_samples 5 --grid_samples 30 \
--max_range_0 0.0 --max_range_1 10.0

RAY_ADDRESS=auto python -u HypOpConstantLR.py \
--dim 10 --max_delay 1 --num_delays 10000 \
--use_delays False --maxiter 10000 \
--cost_function "zakharov" --tol 1e-5 --num_runs 10 \
--filename "Zakharov_UnDelayed1_10" --num_initials 10 \
--num_processes 25 --bayesian_samples 5 --grid_samples 30 \
--max_range_0 0.0 --max_range_1 10.0

RAY_ADDRESS=auto python -u HypOpConstantLR.py \
--dim 100 --max_delay 1 --num_delays 10000 \
--use_delays False --maxiter 10000 \
--cost_function "zakharov" --tol 1e-5 --num_runs 10 \
--filename "Zakharov_UnDelayed1_100" --num_initials 10 \
--num_processes 25 --bayesian_samples 5 --grid_samples 30 \
--max_range_0 0.0 --max_range_1 10.0

RAY_ADDRESS=auto python -u HypOpConstantLR.py \
--dim 1000 --max_delay 1 --num_delays 10000 \
--use_delays False --maxiter 10000 \
--cost_function "zakharov" --tol 1e-5 --num_runs 10 \
--filename "Zakharov_UnDelayed1_1000" --num_initials 10 \
--num_processes 25 --bayesian_samples 5 --grid_samples 30 \
--max_range_0 0.0 --max_range_1 10.0

ray stop
python add_hyperparam_data.py

conda deactivate
