#!/bin/bash
set -e

# =========================
# CONFIG
# =========================
TASK_NAME=bimanual_lift_tray

# Load paths from paths.py
RAW_ROOT=$(python3 paths.py RAW_ROOT)
ZARR_ROOT=$(python3 paths.py ZARR_ROOT)
ZARR_OUT=${ZARR_ROOT}/peract2_single_task/${TASK_NAME}

SCRIPT=data_processing/peract2_to_zarr.py

# =========================
# RUN
# =========================
echo "Generating Zarr for task: ${TASK_NAME}"
echo "Output dir: ${ZARR_OUT}"

python ${SCRIPT} \
  --root ${RAW_ROOT} \
  --tgt ${ZARR_OUT} \
  --tasks ${TASK_NAME}

echo "Done."

