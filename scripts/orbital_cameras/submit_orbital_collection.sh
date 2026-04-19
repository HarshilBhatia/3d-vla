#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_orbital_collection.sh
#
# Submit 54-way parallel orbital data collection, then chain a zarr merge
# job that only runs after ALL collection tasks succeed.
#
# Usage (from repo root):
#   bash scripts/rlbench/submit_orbital_collection.sh
#
# Dry run (print sbatch commands without submitting):
#   DRY_RUN=1 bash scripts/rlbench/submit_orbital_collection.sh
#
# Expected outputs:
#   Raw episodes : data/orbital_rollouts/{task}/{group}/episode_*/
#   Merged zarr  : data/orbital_train.zarr
#   Logs         : logs/collect_orbital_<ARRAY_ID>_<IDX>.{out,err}
#                  logs/orbital_zarr_<JOB_ID>.{out,err}
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_DIR="/ocean/projects/cis240058p/hbhatia1/3d-vla"
DRY_RUN="${DRY_RUN:-0}"

COLLECT_SBATCH="${REPO_DIR}/scripts/rlbench/collect_orbital_slurm.sbatch"
MERGE_SBATCH="${REPO_DIR}/scripts/rlbench/merge_orbital_zarr.sbatch"

mkdir -p "${REPO_DIR}/logs" "${REPO_DIR}/data/orbital_rollouts"

echo "============================================"
echo "  Orbital RLBench Data Collection"
echo "  54 array tasks × 30 episodes = 1,620 total"
echo "  DRY_RUN=${DRY_RUN}"
echo "============================================"
echo ""

# ── Submit 54-way collection array job ───────────────────────────────────────
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] sbatch ${COLLECT_SBATCH}"
    COLLECT_JOB_ID="DRY_RUN_ID"
else
    COLLECT_JOB_ID=$(sbatch "${COLLECT_SBATCH}" | awk '{print $NF}')
    echo "[SUBMITTED] Collection array job: ${COLLECT_JOB_ID}"
    echo "            Monitor : squeue -j ${COLLECT_JOB_ID}"
    echo "            Logs    : logs/collect_orbital_${COLLECT_JOB_ID}_<IDX>.out"
fi

echo ""

# ── Submit merge job with afterok dependency ──────────────────────────────────
# afterok on an array job ID waits for ALL 54 elements to complete successfully.
# If any element fails, the merge job is cancelled automatically.
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] sbatch --dependency=afterok:DRY_RUN_ID ${MERGE_SBATCH}"
else
    MERGE_JOB_ID=$(sbatch \
        --dependency="afterok:${COLLECT_JOB_ID}" \
        "${MERGE_SBATCH}" | awk '{print $NF}')
    echo "[SUBMITTED] Merge job: ${MERGE_JOB_ID}"
    echo "            Depends on: all 54 tasks in ${COLLECT_JOB_ID}"
    echo "            Monitor   : squeue -j ${MERGE_JOB_ID}"
    echo "            Log       : logs/orbital_zarr_${MERGE_JOB_ID}.out"
fi

echo ""
echo "============================================"
echo "  All jobs submitted."
echo ""
echo "  Track all jobs  : squeue -u \$USER"
echo "  Cancel array    : scancel ${COLLECT_JOB_ID:-COLLECT_JOB_ID}"
echo ""
echo "  If some tasks fail:"
echo "    1. Check logs/collect_orbital_<ARRAY_ID>_<IDX>.out"
echo "    2. Resubmit failed indices:"
echo "       sbatch --array=<idx1,idx2,...> ${COLLECT_SBATCH}"
echo "    3. Re-trigger merge once all done:"
echo "       sbatch ${MERGE_SBATCH}"
echo ""
echo "  Verify merged zarr:"
echo "    python3 -c \"import zarr; z=zarr.open('data/orbital_train.zarr'); \\"
echo "      print({k: z[k].shape for k in z})\""
echo "============================================"
