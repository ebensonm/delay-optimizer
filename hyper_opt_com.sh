#!/bin/bash

#SBATCH --time=60:00:00   # walltime
#SBATCH --ntasks=50   # number of processor cores (i.e. tasks)
#SBATCH --nodes=10  # number of nodes
#SBATCH --mem-per-cpu=5000M   # memory per CPU core
#SBATCH -J  "Del101"   # job name

#load the modules
module load python/3.8
#add julia module
module load julia/1.4
#add mpi module
module load mpi/openmpi-1.10.7_gnu4.8.5

#now run the script with the appropriate arguments
mpirun python -u hyperparameter_optimization.py --dim 216 \
--max_L 1 --num_delays 250 --use_delays True --maxiter 250 \
--optimizer_name "Adam" --loss_name "Combustion" --tol 1e-4 \
--max_evals 200 --symmetric_delays True --constant_learning_rate False \
--num_tests 1 --filename "Combustion_Test_10" --num_test_initials 1 \
--logging False --vary_percent 0.3 --hyper_minimize "loss"
