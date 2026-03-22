#!/bin/bash
#SBATCH --job-name=E3-synth-bias
#SBATCH --partition=shubhamlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/e3_synthetic_bias-%j.out
#SBATCH --error=logs/e3_synthetic_bias-%j.err
#SBATCH --nodelist=grogu-2-12

# E3: Synthetic rotation bias — with ΔM (bias injection needs code)
set -e
REPO_DIR=/home/lzaceria/mscv/3dvla/3d-vla
echo "Job started at: $(date)"
echo "Experiment: E3 synthetic bias (ΔM; R_bias injection when implemented)"
cd "$REPO_DIR"
source /home/lzaceria/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa
export USER_NAME=LUQMAN
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export USE_ROPE_DELTA_M=true
export WANDB_NAME=e3-synthetic-bias

nvidia-smi
bash scripts/rlbench/train_peract2.sh
echo "Job finished at: $(date)"
