#!/bin/bash
#SBATCH --job-name=E9-perhead
#SBATCH --partition=shubhamlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/e9_perhead-%j.out
#SBATCH --error=logs/e9_perhead-%j.err
#SBATCH --nodelist=grogu-2-12

# E9: Per-head vs shared — Per-head ΔM (same as shared until per_head wired)
set -e
REPO_DIR=/home/lzaceria/mscv/3dvla/3d-vla
echo "Job started at: $(date)"
echo "Experiment: E9 perhead"
cd "$REPO_DIR"
source /home/lzaceria/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa
export USER_NAME=LUQMAN
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export USE_ROPE_DELTA_M=true
export ROPE_LAMBDA_REG=0.0
export WANDB_NAME=e9-perhead

nvidia-smi
bash scripts/rlbench/train_peract2.sh
echo "Job finished at: $(date)"
