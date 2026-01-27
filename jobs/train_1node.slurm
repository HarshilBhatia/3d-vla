#!/bin/bash
#SBATCH --job-name=train_peract2
#SBATCH --output=logs/train_peract2_%j.out
#SBATCH --error=logs/train_peract2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --partition=shubhamlong
#SBATCH --nodelist=grogu-1-34 

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Change to the project directory
cd /home/harshilb/3d_flowmatch_actor

# Activate conda environment (uncomment and adjust if needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

# Run the training script
bash scripts/rlbench/train_peract2.sh

echo "Job finished at: $(date)"
