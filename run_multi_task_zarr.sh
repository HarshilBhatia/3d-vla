#!/bin/bash
set -e

# =========================
# CONFIG
# =========================
# Option 1: Specify multiple tasks (comma-separated, no spaces)
TASK_LIST="bimanual_lift_tray,bimanual_push_box"

# Option 2: Leave empty to use ALL default tasks (13 tasks)
# TASK_LIST=""

# Load paths from paths.py
RAW_ROOT=$(python3 paths.py RAW_ROOT)
ZARR_ROOT=$(python3 paths.py ZARR_ROOT)
ZARR_OUT=${ZARR_ROOT}/${TASK_LIST}

SCRIPT=data_processing/peract2_to_zarr.py

# =========================
# RUN
# =========================
if [ -z "$TASK_LIST" ]; then
    echo "Generating Zarr for ALL default tasks"
    python ${SCRIPT} \
      --root ${RAW_ROOT} \
      --tgt ${ZARR_OUT}
else
    echo "Generating Zarr for tasks: ${TASK_LIST}"
    python ${SCRIPT} \
      --root ${RAW_ROOT} \
      --tgt ${ZARR_OUT} \
      --tasks ${TASK_LIST}
fi

echo "Output dir: ${ZARR_OUT}"
echo "Done."
