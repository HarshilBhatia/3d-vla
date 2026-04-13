#!/bin/bash
#SBATCH --job-name=3dvla
#SBATCH --partition=shubhamlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

# Print some info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Change to the project directory
cd /home/lzaceria/mscv/3dvla/3d-vla

# Activate conda environment
source /home/lzaceria/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

# Set environment variables if needed
export USER_NAME=LUQMAN  # For paths.py configuration
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print GPU info
echo "GPU Info:"
nvidia-smi

# Run the training script
echo "Starting training..."
bash scripts/rlbench/train_peract2.sh

echo "Job finished at: $(date)"
