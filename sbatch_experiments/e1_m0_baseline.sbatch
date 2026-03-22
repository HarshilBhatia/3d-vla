#!/bin/bash
#SBATCH --job-name=E1-M0-baseline
#SBATCH --partition=shubhamlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/e1_m0_baseline-%j.out
#SBATCH --error=logs/e1_m0_baseline-%j.err

# E1: Does ΔM help? — M0: Baseline (no RoPE ΔM)
set -e
REPO_DIR=/home/lzaceria/mscv/3dvla/3d-vla
echo "Job started at: $(date)"
echo "Experiment: E1 M0 baseline (no ΔM)"
cd "$REPO_DIR"
source /home/lzaceria/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa
export USER_NAME=LUQMAN
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export USE_ROPE_DELTA_M=false
export WANDB_NAME=e1-m0-baseline

nvidia-smi
bash scripts/rlbench/train_peract2.sh
echo "Job finished at: $(date)"
