#!/bin/bash

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=50   # number of processor cores (i.e. tasks)
#SBATCH --nodes=5  # number of nodes
#SBATCH --mem-per-cpu=5000M   # memory per CPU core
#SBATCH -J  "RastDel"   # job name

#load the modules
module load python/3.8
module load miniconda3/4.6
module load mpi/openmpi-1.10.7_gnu4.8.5
source activate HyperOpt

#now run the script with the appropriate arguments
mpirun python -u hyperparameter_optimization.py --dim 2 \
--max_L 2 --num_delays 1000 --use_delays True --maxiter 500 \
--optimizer_name "Adam" --loss_name "Ackley" --tol 1e-5 \
--max_evals 10 --symmetric_delays True --constant_learning_rate False \
--num_tests 1 --filename "Simple_Test" --num_test_initials 5

#close and delete the environment
conda deactivate