#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=50   # number of processor cores (i.e. tasks)
#SBATCH --nodes=5  # number of nodes
#SBATCH --mem-per-cpu=8000M   # memory per CPU core
#SBATCH -J  "DelTOff"   # job name

module load miniconda3/4.6
source activate HypOpt

ray start --head
sleep 10
ray status

RAY_ADDRESS=auto python -u HypOpConstantLR.py \
--dim 1000 --max_delay 1 --num_delays 1000 \
--use_delays True --maxiter 5000 \
--cost_function "ackley" --tol 1e-5 --num_runs 20 \
--filename "Ackley_Delayed_TF_1000" --num_initials 20 \
--num_processes 2 --bayesian_samples 5 --grid_samples 15 \
--max_range_0 0.0 --max_range_1 4.0

python add_hyperparam_data.py


RAY_ADDRESS=auto python -u HypOpConstantLR.py \
--dim 1000 --max_delay 1 --num_delays 1000 \
--use_delays True --maxiter 5000 \
--cost_function "zakharov" --tol 1e-5 --num_runs 20 \
--filename "Zakharov_Delayed_TF_1000" --num_initials 20 \
--num_processes 2 --bayesian_samples 5 --grid_samples 20 \
--max_range_0 0.0 --max_range_1 7.0

python add_hyperparam_data.py

RAY_ADDRESS=auto python -u HypOpConstantLR.py \
--dim 1000 --max_delay 1 --num_delays 1000 \
--use_delays True --maxiter 5000 \
--cost_function "ackley" --tol 1e-5 --num_runs 20 \
--filename "Ackley_Delayed_TF_1000" --num_initials 20 \
--num_processes 2 --bayesian_samples 5 --grid_samples 15 \
--max_range_0 0.0 --max_range_1 4.0

python add_hyperparam_data.py


RAY_ADDRESS=auto python -u HypOpConstantLR.py \
--dim 1000 --max_delay 1 --num_delays 1000 \
--use_delays True --maxiter 5000 \
--cost_function "zakharov" --tol 1e-5 --num_runs 20 \
--filename "Zakharov_Delayed_TF_1000" --num_initials 20 \
--num_processes 2 --bayesian_samples 5 --grid_samples 20 \
--max_range_0 0.0 --max_range_1 7.0

python add_hyperparam_data.py

RAY_ADDRESS=auto python -u HypOpConstantLR.py \
--dim 1000 --max_delay 1 --num_delays 1000 \
--use_delays True --maxiter 5000 \
--cost_function "ackley" --tol 1e-5 --num_runs 20 \
--filename "Ackley_Delayed_TF_1000" --num_initials 20 \
--num_processes 2 --bayesian_samples 5 --grid_samples 15 \
--max_range_0 0.0 --max_range_1 4.0

python add_hyperparam_data.py


RAY_ADDRESS=auto python -u HypOpConstantLR.py \
--dim 1000 --max_delay 1 --num_delays 1000 \
--use_delays True --maxiter 5000 \
--cost_function "zakharov" --tol 1e-5 --num_runs 20 \
--filename "Zakharov_Delayed_TF_1000" --num_initials 20 \
--num_processes 2 --bayesian_samples 5 --grid_samples 20 \
--max_range_0 0.0 --max_range_1 7.0

python add_hyperparam_data.py

RAY_ADDRESS=auto python -u HypOpConstantLR.py \
--dim 1000 --max_delay 1 --num_delays 1000 \
--use_delays True --maxiter 5000 \
--cost_function "ackley" --tol 1e-5 --num_runs 20 \
--filename "Ackley_Delayed_TF_1000" --num_initials 20 \
--num_processes 2 --bayesian_samples 5 --grid_samples 15 \
--max_range_0 0.0 --max_range_1 4.0

python add_hyperparam_data.py


RAY_ADDRESS=auto python -u HypOpConstantLR.py \
--dim 1000 --max_delay 1 --num_delays 1000 \
--use_delays True --maxiter 5000 \
--cost_function "zakharov" --tol 1e-5 --num_runs 20 \
--filename "Zakharov_Delayed_TF_1000" --num_initials 20 \
--num_processes 2 --bayesian_samples 5 --grid_samples 20 \
--max_range_0 0.0 --max_range_1 7.0

python add_hyperparam_data.py

RAY_ADDRESS=auto python -u HypOpConstantLR.py \
--dim 1000 --max_delay 1 --num_delays 1000 \
--use_delays True --maxiter 5000 \
--cost_function "ackley" --tol 1e-5 --num_runs 20 \
--filename "Ackley_Delayed_TF_1000" --num_initials 20 \
--num_processes 2 --bayesian_samples 5 --grid_samples 15 \
--max_range_0 0.0 --max_range_1 4.0

python add_hyperparam_data.py


RAY_ADDRESS=auto python -u HypOpConstantLR.py \
--dim 1000 --max_delay 1 --num_delays 1000 \
--use_delays True --maxiter 5000 \
--cost_function "zakharov" --tol 1e-5 --num_runs 20 \
--filename "Zakharov_Delayed_TF_1000" --num_initials 20 \
--num_processes 2 --bayesian_samples 5 --grid_samples 20 \
--max_range_0 0.0 --max_range_1 7.0

python add_hyperparam_data.py


ray stop

conda deactivate
