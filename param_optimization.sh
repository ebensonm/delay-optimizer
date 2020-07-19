#!/bin/bash

#SBATCH --time=4:00:00   # walltime
#SBATCH --ntasks=50   # number of processor cores (i.e. tasks)
#SBATCH --nodes=10  # number of nodes
#SBATCH --mem-per-cpu=4000M   # memory per CPU core
#SBATCH -J  "2A_del Test"   # job name

#add the right thing to the file path
export PATH="$HOME/.local/bin:$PATH"

#load the modules
module load python/3.8
module load mpi/openmpi-1.10.7_gnu4.8.5

#now run the script
mpirun python -u param_optimization.py
