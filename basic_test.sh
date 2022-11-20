#!/bin/bash

#SBATCH --time=6:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=5000M   # memory per CPU core
#SBATCH -J  "10Test"   # job name

module load miniconda3/4.6
source activate HypOpt

ray start --head
sleep 10
ray status
RAY_ADDRESS=auto python -u HypOpConstantLR.py --dim 10 --max_delay 1 --num_delays 10000 --use_delays False --maxiter 1000 \
--cost_function "rosenbrock" --tol 1e-5 --num_runs 5 \
--filename "Rosenbrock_Basic" --num_initials 5 \
--num_processes 7 --bayesian_samples 5 --grid_samples 10 \
--ranges_0 0.0 --ranges_1 3.0


ray stop

conda deactivate
