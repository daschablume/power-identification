#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -J hr_nn
#SBATCH -o logs_uppmax/hr_nn.out
#SBATCH -e logs_uppmax/hr_nn.err
#SBATCH --gres=gpu:1

STORAGE_PJ=uppmax2024-2-13
ENV_DIR=/proj/${STORAGE_PJ}/hapham/envs/power-identification
PROJECT_DIR=/proj/${STORAGE_PJ}/hapham/power-identification

conda activate ${ENV_DIR}
cd ${PROJECT_DIR}

# Change path to python script
python experiments/classic_ml/experiment_hr_nn.py
