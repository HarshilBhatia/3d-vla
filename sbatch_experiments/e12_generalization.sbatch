#!/bin/bash
#SBATCH --job-name=E12-generalization
#SBATCH --partition=shubhamlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/e12_generalization-%j.out
#SBATCH --error=logs/e12_generalization-%j.err
#SBATCH --nodelist=grogu-2-12

# E12: Generalization — Train with ΔM; optional ROTATE_PCD for augmentation
# For test-time rotation eval, use eval pipeline when implemented.
set -e
REPO_DIR=/home/lzaceria/mscv/3dvla/3d-vla
echo "Job started at: $(date)"
echo "Experiment: E12 generalization"
cd "$REPO_DIR"
source /home/lzaceria/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa
export USER_NAME=LUQMAN
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export USE_ROPE_DELTA_M=true
export WANDB_NAME=e12-generalization
# Optional: export ROTATE_PCD=true ROTATE_ANGLE_DEG=15 ROTATE_AXIS=z

nvidia-smi
bash scripts/rlbench/train_peract2.sh
echo "Job finished at: $(date)"
