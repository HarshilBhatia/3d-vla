#!/bin/bash
#SBATCH --job-name=generate_zarr_all
#SBATCH --partition=shubhamlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/generate_zarr_all-%j.out
#SBATCH --error=logs/generate_zarr_all-%j.err

# Generate Zarr dataset for ALL tasks found in peract2_raw
set -e
REPO_DIR=/home/lzaceria/mscv/3dvla/3d-vla
echo "Job started at: $(date)"
echo "Generating Zarr dataset for ALL tasks..."
cd "$REPO_DIR"
source /home/lzaceria/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa
export USER_NAME=LUQMAN

# This will scan peract2_raw for tasks in DEFAULT_TASKS list and generate train.zarr/val.zarr in Peract2_zarr/all
python3 data_processing/peract2_to_zarr.py --tgt Peract2_zarr/all

echo "Job finished at: $(date)"
