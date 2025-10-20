#!/bin/bash
#SBATCH --job-name=oasis_train           # Name of your job
#SBATCH --partition=comp3710             # Partition to submit to (as per your previous scripts)
#SBATCH --gres=gpu:a100:1                # Request 1 A100 GPU
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --mem=16G                        # Memory per node
#SBATCH --time=02:00:00                  # Walltime (hh:mm:ss)
#SBATCH -o oasis_train.out               # Standard output file
#SBATCH -e oasis_train.err               # Standard error file

# Load your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

# Run your training script
python train_oasis.py   