#!/bin/bash

#SBATCH --time=0:30:00   # walltime
#SBATCH --ntasks=50   # number of processor cores (i.e. tasks)
#SBATCH --nodes=10  # number of nodes
#SBATCH --mem-per-cpu=5000M   # memory per CPU core
#SBATCH -J  "CombustionDel"   # job name

#load the modules
module load python/3.8
module load miniconda3/4.6
module load mpi/openmpi-1.10.7_gnu4.8.5
source activate HyperOpt

#now run the script with the appropriate arguments
mpirun python -u hyperparameter_optimization.py --dim 216 \
--max_L 1 --num_delays 100 --use_delays True --maxiter 10 \
--optimizer_name "Adam" --loss_name "Combustion" --tol 1e-4 \
--max_evals 1 --symmetric_delays True --constant_learning_rate False \
--num_tests 1 --filename "Combustion_Test" --num_test_initials 1 \
--logging True --vary_percent 0.1 --hyper_minimize "distance"

#close and delete the environment
conda deactivate
