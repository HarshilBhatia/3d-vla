#!/bin/bash
#SBATCH --job-name=peract_zarr_rot
#SBATCH --partition=shubhamlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/peract_zarr_rot10_transform-%j.out
#SBATCH --error=logs/peract_zarr_rot10_transform-%j.err

# Prepare RLBench PerAct zarr with rotation (all 3 axes) and translation.
# Default: 10 deg around Z; override with env or sbatch --export.
# Usage:
#   sbatch sbatch_experiments/peract_zarr_rot10_transform.sbatch
#   ROTATE_X=5 ROTATE_Y=0 ROTATE_Z=10 TRANSLATE="0.01,0,0" sbatch ...
set -e
REPO_DIR="${REPO_DIR:-/home/lzaceria/mscv/3dvla/3d-vla}"
PERACT_ROOT="${PERACT_ROOT:-$REPO_DIR/peract_raw/Peract_packaged/}"
PERACT_ZARR="${PERACT_ZARR:-$REPO_DIR/Peract_zarr_rot10deg}"

# Rotation in degrees (default: 10 around Z only)
ROTATE_X="${ROTATE_X:-0}"
ROTATE_Y="${ROTATE_Y:-0}"
ROTATE_Z="${ROTATE_Z:-10}"
TRANSLATE="${TRANSLATE:-0,0,0}"

echo "Job started at: $(date)"
echo "PerAct zarr (transform): root=$PERACT_ROOT tgt=$PERACT_ZARR"
echo "  rotate_x=${ROTATE_X}° rotate_y=${ROTATE_Y}° rotate_z=${ROTATE_Z}° translate=${TRANSLATE}"
cd "$REPO_DIR"
source "${CONDA_PREFIX:-/home/lzaceria/miniconda3}/etc/profile.d/conda.sh"
conda activate 3dfa
export USER_NAME="${USER_NAME:-LUQMAN}"

mkdir -p "$PERACT_ZARR" logs
python -m data_processing.peract_to_zarr \
  --root "$PERACT_ROOT" \
  --tgt "$PERACT_ZARR" \
  --rotate_x_deg "$ROTATE_X" \
  --rotate_y_deg "$ROTATE_Y" \
  --rotate_z_deg "$ROTATE_Z" \
  --translate "$TRANSLATE"

echo "Job finished at: $(date)"
