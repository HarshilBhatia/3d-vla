#!/usr/bin/env bash
# Claim a run directory for this Slurm job; exit 3 if a different, still-running job owns it.
set -euo pipefail
RUN_DIR="${1:?usage: claim_run.sh <run dir>}"
JOB="${SLURM_JOB_ID:?claim_run.sh must run inside a Slurm job}"
mkdir -p "$RUN_DIR"
OWNER_FILE="$RUN_DIR/.owner_job"
if [[ -f "$OWNER_FILE" ]]; then
    owner=$(cat "$OWNER_FILE")
    if [[ "$owner" != "$JOB" ]] && squeue -h -j "$owner" -t RUNNING 2>/dev/null | grep -q .; then
        echo "[claim] $RUN_DIR is owned by running job $owner; job $JOB will not train"
        exit 3
    fi
fi
echo "$JOB" > "$OWNER_FILE"
echo "[claim] job $JOB owns $RUN_DIR"
