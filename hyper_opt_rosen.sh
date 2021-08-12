#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=50   # number of processor cores (i.e. tasks)
#SBATCH --nodes=5  # number of nodes
#SBATCH --mem-per-cpu=7000M   # memory per CPU core
#SBATCH -J  "RosenDel"   # job name

#load the modules
module load miniconda3/4.6
module load mpi/openmpi-1.10.7_gnu4.8.5
source activate rosenbrock


mpirun python -u hyperparameter_optimization.py --dim 2 \
--max_L 1 --num_delays 1000 --use_delays True --maxiter 3000 \
--optimizer_name "Adam" --loss_name "Rosenbrock" --tol 1e-5 \
--max_evals 200 --symmetric_delays True --constant_learning_rate False \
--num_tests 1 --filename "Delayed_Rosen_2d" --num_test_initials 5

#now run the script with the appropriate arguments
mpirun python -u hyperparameter_optimization.py --dim 10 \
--max_L 1 --num_delays 1000 --use_delays True --maxiter 3000 \
--optimizer_name "Adam" --loss_name "Rosenbrock" --tol 1e-5 \
--max_evals 200 --symmetric_delays True --constant_learning_rate False \
--num_tests 1 --filename "Delayed_Rosen_10d" --num_test_initials 5

#now run the script with the appropriate arguments
mpirun python -u hyperparameter_optimization.py --dim 100 \
--max_L 1 --num_delays 1000 --use_delays True --maxiter 3000 \
--optimizer_name "Adam" --loss_name "Rosenbrock" --tol 1e-5 \
--max_evals 200 --symmetric_delays True --constant_learning_rate False \
--num_tests 1 --filename "Delayed_Rosen_100d" --num_test_initials 5

#now run the script with the appropriate arguments
mpirun python -u hyperparameter_optimization.py --dim 1000 \
--max_L 1 --num_delays 1000 --use_delays True --maxiter 3000 \
--optimizer_name "Adam" --loss_name "Rosenbrock" --tol 1e-5 \
--max_evals 200 --symmetric_delays True --constant_learning_rate False \
--num_tests 1 --filename "Delayed_Rosen_1000d" --num_test_initials 5

#now run the script with the appropriate arguments
mpirun python -u hyperparameter_optimization.py --dim 10000 \
--max_L 1 --num_delays 1000 --use_delays True --maxiter 3000 \
--optimizer_name "Adam" --loss_name "Rosenbrock" --tol 1e-5 \
--max_evals 200 --symmetric_delays True --constant_learning_rate False \
--num_tests 1 --filename "Delayed_Rosen_10000d" --num_test_initials 5

#close and delete the environment
conda deactivate
