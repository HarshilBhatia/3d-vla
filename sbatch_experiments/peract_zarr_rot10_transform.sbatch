#!/bin/bash
#SBATCH --job-name=peract_zarr_rot
#SBATCH --partition=shubhamlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j-peract_zarr_rot10_transform.out
#SBATCH --error=logs/%j-peract_zarr_rot10_transform.err

# Build PerAct zarr from raw data that was COLLECTED with camera rig rotated 10°
# (no post-hoc image rotation). For clean rot10 views you must first collect demos
# with RLBench using --camera_rig_rotation_deg=10 and package them to the same
# layout as Peract_packaged (train/, val/, task+var/ with .dat files).
#
# Usage:
#   1) Collect rotated data (see scripts/rlbench/README_RLBench.md):
#        python RLBench/tools/dataset_generator.py --save_path .../peract_raw_rot10 ...
#        --camera_rig_rotation_deg 10
#      Then package to Peract_packaged layout under peract_raw_rot10/ if needed.
#   2) sbatch sbatch_experiments/peract_zarr_rot10_transform.sbatch
#   Override: PERACT_ROOT_ROT10=/path/to/rot10/packaged PERACT_ZARR=/path/to/out sbatch ...
set -e
REPO_DIR="${REPO_DIR:-/home/lzaceria/mscv/3dvla/3d-vla}"
PERACT_ROOT_ROT10="${PERACT_ROOT_ROT10:-$REPO_DIR/peract_raw_rot10/Peract_packaged/}"
PERACT_ZARR="${PERACT_ZARR:-$REPO_DIR/Peract_zarr_rot10deg}"

echo "Job started at: $(date)"
echo "PerAct zarr (from camera-rotated data, no post-hoc transform):"
echo "  root=$PERACT_ROOT_ROT10"
echo "  tgt=$PERACT_ZARR"

if [[ ! -d "$PERACT_ROOT_ROT10" ]]; then
  echo "[ERROR] Rot10 raw root not found: $PERACT_ROOT_ROT10"
  echo "Collect data with RLBench using --camera_rig_rotation_deg 10 and package to this path (same layout as Peract_packaged). See scripts/rlbench/README_RLBench.md"
  exit 1
fi
if [[ ! -d "$PERACT_ROOT_ROT10/train" ]] || [[ ! -d "$PERACT_ROOT_ROT10/val" ]]; then
  echo "[ERROR] Rot10 root must contain train/ and val/ (same layout as Peract_packaged). Got: $PERACT_ROOT_ROT10"
  exit 1
fi

cd "$REPO_DIR"
source "${CONDA_PREFIX:-/home/lzaceria/miniconda3}/etc/profile.d/conda.sh"
conda activate 3dfa
pip install blosc -q
export USER_NAME="${USER_NAME:-LUQMAN}"

mkdir -p "$PERACT_ZARR" logs
python -m data_processing.peract_to_zarr \
  --root "$PERACT_ROOT_ROT10" \
  --tgt "$PERACT_ZARR" \
  --overwrite

echo "Job finished at: $(date)"
