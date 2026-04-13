#!/bin/bash
#SBATCH --job-name=peract_zarr_perturbed
#SBATCH --partition=all
#SBATCH --exclude=grogu-1-34
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%j-peract_zarr_perturbed.out
#SBATCH --error=logs/%j-peract_zarr_perturbed.err

# Build zarr from all 18-task Peract_packaged_perturbed/ directory.
# Run AFTER all peract_collect.sbatch jobs complete (use --dependency=afterok:... in submit script).
#
# Usage:
#   sbatch sbatch_experiments/peract_zarr_perturbed.sbatch
#   PERACT_ZARR_OUT=/path/to/zarr sbatch ...

set -eo pipefail
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND (exit $?)" >&2' ERR

REPO_DIR="${REPO_DIR:-/home/lzaceria/mscv/3dvla/3d-vla}"
PERACT_PACKAGED_IN="${PERACT_PACKAGED_IN:-$REPO_DIR/Peract_packaged_perturbed}"
PERACT_ZARR_OUT="${PERACT_ZARR_OUT:-$REPO_DIR/Peract_zarr_perturbed}"

echo "=========================================="
echo "Job started at: $(date)"
echo "  INPUT:  $PERACT_PACKAGED_IN"
echo "  OUTPUT: $PERACT_ZARR_OUT"
echo "=========================================="

cd "$REPO_DIR"
mkdir -p logs "$PERACT_ZARR_OUT"

CONDA_BASE="${CONDA_BASE:-/home/lzaceria/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate 3dfa

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"
pip install blosc 2>&1 | tail -1

# Verify all 18 tasks are present
PERACT_18="place_cups close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap"
echo "Checking all 18 tasks are present in $PERACT_PACKAGED_IN/train/ ..."
missing=()
for task in $PERACT_18; do
  n=$(ls "$PERACT_PACKAGED_IN/train/" 2>/dev/null | grep "^${task}+" | wc -l)
  if [[ "$n" -eq 0 ]]; then
    missing+=("$task")
  fi
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "[WARN] Missing tasks in train/: ${missing[*]}"
  echo "  Proceeding with available tasks only."
else
  echo "  All 18 tasks present."
fi

# NOTE: trailing slash on --root is required
echo "Building zarr -> $PERACT_ZARR_OUT ..."
python -m data_processing.peract_to_zarr \
  --root "$PERACT_PACKAGED_IN/" \
  --tgt "$PERACT_ZARR_OUT" \
  --overwrite

echo "Done at: $(date)"
echo "Zarr: $PERACT_ZARR_OUT/train.zarr  $PERACT_ZARR_OUT/val.zarr"
