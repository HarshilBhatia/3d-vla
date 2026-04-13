#!/bin/bash
set -e

# =========================
# CONFIG
# =========================
# Specify tasks as a comma-separated list, e.g., "bimanual_lift_tray,bimanual_push_box"
TASKS="bimanual_lift_tray"

# Load paths from paths.py
RAW_ROOT=$(python3 paths.py RAW_ROOT)
ZARR_ROOT=$(python3 paths.py ZARR_ROOT)


# =========================
# PROCESS EACH TASK
# =========================
IFS=',' read -ra ADDR <<< "$TASKS"
for TASK in "${ADDR[@]}"; do
    echo "------------------------------------------------------------------"
    echo "PROCESSING TASK: ${TASK}"
    echo "------------------------------------------------------------------"

    # 1. Download Raw Data
    echo "[STEP 1] Checking/Downloading raw data..."
    python scripts/rlbench/download_peract2.py --root ${RAW_ROOT} --tasks ${TASK}

    # 2. Generate Zarr
    ZARR_OUT=${ZARR_ROOT}/${TASK}
    echo "[STEP 2] Generating Zarr in: ${ZARR_OUT}"
    
    # Clean up existing Zarr for this specific task
    rm -rf ${ZARR_OUT}/train.zarr ${ZARR_OUT}/val.zarr
    mkdir -p ${ZARR_OUT}

    python data_processing/peract2_to_zarr.py \
      --root ${RAW_ROOT} \
      --tgt ${ZARR_OUT} \
      --tasks ${TASK}

    echo "Done with ${TASK}."
done

echo "------------------------------------------------------------------"
echo "All tasks processed. Zarr files are in subfolders of ${ZARR_ROOT}"
