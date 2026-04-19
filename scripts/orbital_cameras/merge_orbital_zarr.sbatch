#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# merge_orbital_zarr.sbatch
#
# Merge all raw orbital episode data into a single train.zarr.
# Reads data/orbital_rollouts/{task}/{group}/episode_*/ (all 54 combos)
# and writes data/orbital_train.zarr.
#
# No GPU needed — pure numpy/zarr processing.
#
# Typically submitted automatically via submit_orbital_collection.sh with
# --dependency=afterok:<collection_array_job_id>.
#
# Manual submission (after verifying all collection jobs succeeded):
#   cd /ocean/projects/cis240058p/hbhatia1/3d-vla
#   sbatch scripts/rlbench/merge_orbital_zarr.sbatch
#
# Verify output:
#   python3 -c "import zarr; z=zarr.open('data/orbital_train.zarr'); \
#               print({k: z[k].shape for k in z})"
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=orbital_zarr
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=04:00:00
#SBATCH --output=logs/orbital_zarr_%j.out
#SBATCH --error=logs/orbital_zarr_%j.err

set -euo pipefail
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND (exit $?)" >&2' ERR

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="/ocean/projects/cis240058p/hbhatia1/3d-vla"
CONTAINER="/ocean/projects/cis240058p/hbhatia1/containers/3dfa-sandbox"
DATA_ROOT="${REPO_DIR}/data/orbital_rollouts"
OUTPUT_ZARR="${REPO_DIR}/data/orbital_train.zarr"

echo "========================================"
echo "  Zarr merge job ${SLURM_JOB_ID}"
echo "  Node   : $(hostname)"
echo "  Started: $(date)"
echo "  Root   : ${DATA_ROOT}"
echo "  Output : ${OUTPUT_ZARR}"
echo "========================================"

mkdir -p "${REPO_DIR}/logs"

apptainer exec \
    --bind "${REPO_DIR}:${REPO_DIR}" \
    "${CONTAINER}" \
    python3 "${REPO_DIR}/data_processing/orbital_to_zarr.py" \
        --root      "${DATA_ROOT}" \
        --out       "${OUTPUT_ZARR}" \
        --overwrite

echo "========================================"
echo "  Done: $(date)"
echo "  Zarr: ${OUTPUT_ZARR}"
echo "========================================"
