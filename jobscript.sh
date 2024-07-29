#!/bin/bash
#SBATCH --time=0-24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output="%j.out"
#SBATCH --partition=rleap_gpu_24gb
#SBATCH --job-name="pong_supervised_gnn"

# Initialize Conda
source /work/rleap1/rishabh.bhatia/miniconda3/etc/profile.d/conda.sh

# Activate your Conda environment (replace 'train' with the actual environment name)
conda activate train

# Run your Python script
python -m games.pong.run_supervised_cnn


# Deactivate the Conda environment
conda deactivate