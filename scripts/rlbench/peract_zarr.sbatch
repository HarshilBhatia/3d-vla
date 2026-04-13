#!/bin/bash
#SBATCH --job-name=peract_zarr
#SBATCH --partition=shubhamlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/peract_zarr-%j.out
#SBATCH --error=logs/peract_zarr-%j.err

# Prepare RLBench PerAct (unimanual, 18 tasks) zarr with no transform.
# Requires PerAct raw data at PERACT_ROOT (e.g. Peract_packaged/ with train/, val/).
set -e
REPO_DIR="${REPO_DIR:-/home/lzaceria/mscv/3dvla/3d-vla}"
PERACT_ROOT="${PERACT_ROOT:-$REPO_DIR/peract_raw/Peract_packaged/}"
PERACT_ZARR="${PERACT_ZARR:-$REPO_DIR/Peract_zarr}"

echo "Job started at: $(date)"
echo "PerAct zarr (no transform): root=$PERACT_ROOT tgt=$PERACT_ZARR"
cd "$REPO_DIR"
source "${CONDA_PREFIX:-/home/lzaceria/miniconda3}/etc/profile.d/conda.sh"
conda activate 3dfa
export USER_NAME="${USER_NAME:-LUQMAN}"

mkdir -p "$PERACT_ZARR" logs
python -m data_processing.peract_to_zarr --root "$PERACT_ROOT" --tgt "$PERACT_ZARR"

echo "Job finished at: $(date)"
