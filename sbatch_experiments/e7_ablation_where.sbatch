#!/bin/bash
#SBATCH --job-name=E7-ablation-where
#SBATCH --partition=shubhamlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/e7_ablation_where-%j.out
#SBATCH --error=logs/e7_ablation_where-%j.err
#SBATCH --nodelist=grogu-2-12

# E7: Ablation where ΔM is applied — current (RoPE ΔM)
set -e
REPO_DIR=/home/lzaceria/mscv/3dvla/3d-vla
echo "Job started at: $(date)"
echo "Experiment: E7 ablation where (Q-only/K-only need code)"
cd "$REPO_DIR"
source /home/lzaceria/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa
export USER_NAME=LUQMAN
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export USE_ROPE_DELTA_M=true
export WANDB_NAME=e7-ablation-where

nvidia-smi
bash scripts/rlbench/train_peract2.sh
echo "Job finished at: $(date)"
