#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -J ua_sklearn
#SBATCH -o logs_uppmax/ua_sklearn.out
#SBATCH -e logs_uppmax/ua_sklearn.err
#SBATCH --gres=gpu:1

STORAGE_PJ=uppmax2024-2-13
ENV_DIR=/proj/${STORAGE_PJ}/hapham/envs/power-identification
PROJECT_DIR=/proj/${STORAGE_PJ}/hapham/power-identification

conda activate ${ENV_DIR}
cd ${PROJECT_DIR}

# Change path to python script
python experiments/classic_ml/experiment_ua_sklearn.py
