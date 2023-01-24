#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=50   # number of processor cores (i.e. tasks)
#SBATCH --nodes=5  # number of nodes
#SBATCH --mem-per-cpu=8000M   # memory per CPU core
#SBATCH -J  "VaryLR"   # job name

module load miniconda3/4.6
source activate HypOpt

ray start --head
sleep 10
ray status

RAY_ADDRESS=auto python -u HypOpNonConst.py \
--dim 1000 --max_delay 1 --num_delays 10000 \
--use_delays False --maxiter 5000 \
--cost_function "ackley" --tol 1e-5 --num_runs 100 \
--filename "Ackley_UnDelayed_1000_inv" --num_initials 20 \
--num_processes 49 --bayesian_samples 30 \
--lr_type "inv" --p_1 0.0 --p_2 1.0 \
--gamma_1 0.0 --gamma_2 1.0 \
--max_range_0 0.0 --max_range_1 5.0

RAY_ADDRESS=auto python -u HypOpNonConst.py \
--dim 1000 --max_delay 1 --num_delays 10000 \
--use_delays True --maxiter 5000 \
--cost_function "ackley" --tol 1e-5 --num_runs 100 \
--filename "Ackley_Delayed_1000_inv" --num_initials 20 \
--num_processes 49 --bayesian_samples 30 \
--lr_type "inv" --p_1 0.0 --p_2 1.0 \
--gamma_1 0.0 --gamma_2 1.0 \
--max_range_0 0.0 --max_range_1 5.0

RAY_ADDRESS=auto python -u HypOpNonConst.py \
--dim 1000 --max_delay 1 --num_delays 10000 \
--use_delays False --maxiter 5000 \
--cost_function "zakharov" --tol 1e-5 --num_runs 100 \
--filename "Zakharov_UnDelayed_1000_inv" --num_initials 20 \
--num_processes 49 --bayesian_samples 30 \
--lr_type "inv" --p_1 0.0 --p_2 1.0 \
--gamma_1 0.0 --gamma_2 1.0 \
--max_range_0 0.0 --max_range_1 7.0

RAY_ADDRESS=auto python -u HypOpNonConst.py \
--dim 1000 --max_delay 1 --num_delays 10000 \
--use_delays True --maxiter 5000 \
--cost_function "zakharov" --tol 1e-5 --num_runs 100 \
--filename "Zakharov_Delayed_1000_inv" --num_initials 20 \
--num_processes 49 --bayesian_samples 30 \
--lr_type "inv" --p_1 0.0 --p_2 1.0 \
--gamma_1 0.0 --gamma_2 1.0 \
--max_range_0 0.0 --max_range_1 7.0

RAY_ADDRESS=auto python -u HypOpNonConst.py \
--dim 1000 --max_delay 1 --num_delays 10000 \
--use_delays False --maxiter 5000 \
--cost_function "ackley" --tol 1e-5 --num_runs 100 \
--filename "Ackley_UnDelayed_1000_sin-2" --num_initials 20 \
--num_processes 49 --bayesian_samples 30 \
--lr_type "sin-2" \
--step_1 100 --step_2 1000 \
--max_range_0 0.0 --max_range_1 5.0 \
--min_range_0 0.0 --min_range_1 5.0

RAY_ADDRESS=auto python -u HypOpNonConst.py \
--dim 1000 --max_delay 1 --num_delays 10000 \
--use_delays True --maxiter 5000 \
--cost_function "ackley" --tol 1e-5 --num_runs 100 \
--filename "Ackley_Delayed_1000_sin-2" --num_initials 20 \
--num_processes 49 --bayesian_samples 30 \
--lr_type "sin-2" \
--step_1 100 --step_2 1000 \
--max_range_0 0.0 --max_range_1 5.0 \
--min_range_0 0.0 --min_range_1 5.0

RAY_ADDRESS=auto python -u HypOpNonConst.py \
--dim 1000 --max_delay 1 --num_delays 10000 \
--use_delays False --maxiter 5000 \
--cost_function "zakharov" --tol 1e-5 --num_runs 100 \
--filename "Zakharov_UnDelayed_1000_sin-2" --num_initials 20 \
--num_processes 49 --bayesian_samples 30 \
--lr_type "sin-2" \
--step_1 100 --step_2 1000 \
--max_range_0 0.0 --max_range_1 7.0 \
--min_range_0 0.0 --min_range_1 7.0

RAY_ADDRESS=auto python -u HypOpNonConst.py \
--dim 1000 --max_delay 1 --num_delays 10000 \
--use_delays True --maxiter 5000 \
--cost_function "zakharov" --tol 1e-5 --num_runs 100 \
--filename "Zakharov_Delayed_1000_sin-2" --num_initials 20 \
--num_processes 49 --bayesian_samples 30 \
--lr_type "sin-2" \
--step_1 100 --step_2 1000 \
--max_range_0 0.0 --max_range_1 7.0 \
--min_range_0 0.0 --min_range_1 7.0

RAY_ADDRESS=auto python -u HypOpNonConst.py \
--dim 1000 --max_delay 1 --num_delays 10000 \
--use_delays False --maxiter 5000 \
--cost_function "ackley" --tol 1e-5 --num_runs 100 \
--filename "Ackley_UnDelayed_1000_Tri-2" --num_initials 20 \
--num_processes 49 --bayesian_samples 30 \
--lr_type "tri-2" \
--step_1 100 --step_2 1000 \
--max_range_0 0.0 --max_range_1 5.0 \
--min_range_0 0.0 --min_range_1 5.0

RAY_ADDRESS=auto python -u HypOpNonConst.py \
--dim 1000 --max_delay 1 --num_delays 10000 \
--use_delays True --maxiter 5000 \
--cost_function "ackley" --tol 1e-5 --num_runs 100 \
--filename "Ackley_Delayed_1000_Tri-2" --num_initials 20 \
--num_processes 49 --bayesian_samples 30 \
--lr_type "tri-2" \
--step_1 100 --step_2 1000 \
--max_range_0 0.0 --max_range_1 5.0 \
--min_range_0 0.0 --min_range_1 5.0

RAY_ADDRESS=auto python -u HypOpNonConst.py \
--dim 1000 --max_delay 1 --num_delays 10000 \
--use_delays False --maxiter 5000 \
--cost_function "zakharov" --tol 1e-5 --num_runs 100 \
--filename "Zakharov_UnDelayed_1000_Tri-2" --num_initials 20 \
--num_processes 49 --bayesian_samples 30 \
--lr_type "tri-2" \
--step_1 100 --step_2 1000 \
--max_range_0 0.0 --max_range_1 7.0 \
--min_range_0 0.0 --min_range_1 7.0

RAY_ADDRESS=auto python -u HypOpNonConst.py \
--dim 1000 --max_delay 1 --num_delays 10000 \
--use_delays True --maxiter 5000 \
--cost_function "zakharov" --tol 1e-5 --num_runs 100 \
--filename "Zakharov_Delayed_1000_Tri-2" --num_initials 20 \
--num_processes 49 --bayesian_samples 30 \
--lr_type "tri-2" \
--step_1 100 --step_2 1000 \
--max_range_0 0.0 --max_range_1 7.0 \
--min_range_0 0.0 --min_range_1 7.0


ray stop
python add_hyperparam_data.py

conda deactivate
