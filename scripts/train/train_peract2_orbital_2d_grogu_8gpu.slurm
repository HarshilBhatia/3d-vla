#!/usr/bin/env bash
#SBATCH --partition=shubhamlong
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/harshilb/3dfa_unified/logs/train/slurm-2d-8gpu-%j.out
#SBATCH --error=/home/harshilb/3dfa_unified/logs/train/slurm-2d-8gpu-%j.err

# Fallback for the A6000 memory test. This keeps the SAME global batch of 256:
# 8 GPUs x 32 samples/GPU. Use this only if the 4-GPU / 64-per-GPU job OOMs.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
exec "$REPO_ROOT/scripts/train/train_peract2_orbital_2d_grogu.slurm" "$@"
