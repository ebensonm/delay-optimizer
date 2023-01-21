#!/bin/bash

#SBATCH --time=8:00:00   # walltime
#SBATCH --ntasks=50   # number of processor cores (i.e. tasks)
#SBATCH --nodes=5  # number of nodes
#SBATCH --mem-per-cpu=5000M   # memory per CPU core
#SBATCH -J  "2Test"   # job name

module load miniconda3/4.6
source activate HypOpt

ray start --head
sleep 10
ray status
RAY_ADDRESS=auto python -u HypOpConstantLR.py \
--dim 2 --max_delay 1 --num_delays 10000 \
--use_delays False --maxiter 5000 \
--cost_function "ackley" --tol 1e-5 --num_runs 2 \
--filename "Ackley_UnDelayed_2" --num_initials 2 \
--num_processes 2 --bayesian_samples 5 --grid_samples 2 \
--max_range_0 0.0 --max_range_1 2.0

ray stop
python add_hyperparam_data.py

conda deactivate
