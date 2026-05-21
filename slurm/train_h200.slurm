#!/bin/bash
#SBATCH --job-name=pi05_multicam_keypose
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail
echo "Job started: $(date)  Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

cd /work/nvme/bgkz/harshilb/openpi
export PATH="/u/hbhatia1/.local/bin:$PATH"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_multicam_keypose \
    --exp-name multicam_keypose_h200 \
    --overwrite

echo "Done: $(date)"
