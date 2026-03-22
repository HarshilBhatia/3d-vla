#!/bin/bash
#SBATCH --job-name=lz-dmself
#SBATCH --partition=shubhamlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/lz_dmself-%j.out
#SBATCH --error=logs/lz_dmself-%j.err

# Experiment: lz-dmself (Aggressive Self-Attn ΔM)
set -e
REPO_DIR=/home/lzaceria/mscv/3dvla/3d-vla
echo "Job started at: $(date)"
echo "Experiment: lz-dmself (Aggressive Self-Attn ΔM)"
cd "$REPO_DIR"
source /home/lzaceria/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa
export USER_NAME=LUQMAN
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export USE_ROPE_DELTA_M=true
export ROPE_LAMBDA_REG=0.0
export WANDB_NAME=lz-dmself

nvidia-smi
bash scripts/rlbench/train_peract2.sh
echo "Job finished at: $(date)"
