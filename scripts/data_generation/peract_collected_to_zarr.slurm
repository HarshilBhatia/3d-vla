#!/bin/bash
# Convert self-collected PerAct raw demos to zarr for training.
# CPU-only, no GPU needed.
#
# Usage:
#   sbatch scripts/data_generation/peract_collected_to_zarr.slurm

#SBATCH --job-name=peract_to_zarr
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/peract_to_zarr_%j.out
#SBATCH --error=logs/peract_to_zarr_%j.err
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,14],grogu-3-5,grogu-3-25

set -euo pipefail
ulimit -c 0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

cd /home/harshilb/3d_flowmatch_actor
export PYTHONPATH=/home/harshilb/3d_flowmatch_actor:${PYTHONPATH:-}

RAW=/grogu/datasets/hbhatia/peract_rollouts_new
TGT=/grogu/datasets/hbhatia/peract_rollouts_new_zarr

echo "Converting $RAW → $TGT/train.zarr"

python data/processing/convert_to_zarr/peract_collected_to_zarr.py \
    --root "$RAW" \
    --tgt  "$TGT" \
    --tasks light_bulb_in,stack_blocks,put_money_in_safe \
    --overwrite

echo "Done: $TGT/train.zarr"
