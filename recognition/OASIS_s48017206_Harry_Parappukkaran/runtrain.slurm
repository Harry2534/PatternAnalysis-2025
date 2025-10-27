#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --job-name=3Dtrain_oasis
#SBATCH -o 3Dtrain_oasis.out

# Load conda
conda activate torch

# Run Python script
python 3Dtrain_oasis.py