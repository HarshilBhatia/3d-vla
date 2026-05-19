#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Submit collect_full_mapping.sbatch and save the job ID for monitoring.
#
# Usage (from repo root):
#   bash scripts/orbital_cameras/submit_full_mapping.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_DIR="/home/harshilb/3d_flowmatch_actor"
SBATCH="${REPO_DIR}/scripts/orbital_cameras/collect_full_mapping.sbatch"
JOBID_FILE="${REPO_DIR}/logs/collect_full_mapping.jobid"

mkdir -p "${REPO_DIR}/logs"

JOB_ID=$(sbatch "${SBATCH}" | awk '{print $NF}')
echo "${JOB_ID}" > "${JOBID_FILE}"

echo "[SUBMITTED] Job ID: ${JOB_ID}"
echo "            Monitor : squeue -j ${JOB_ID}"
echo "            Logs    : logs/collect_full_${JOB_ID}_<IDX>.out"
echo ""
echo "To monitor and auto-resubmit failures, add this cron entry:"
echo "  crontab -e"
echo "  */30 * * * * cd ${REPO_DIR} && bash scripts/orbital_cameras/monitor_and_resubmit.sh >> logs/monitor_resubmit.log 2>&1"
