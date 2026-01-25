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

# Set up CoppeliaSim environment
export COPPELIASIM_ROOT=/home/lzaceria/mscv/3dvla/3d-vla/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

# Use the full path to the python executable in the 3dfa environment
PYTHON_EXE=/home/lzaceria/miniconda3/envs/3dfa/bin/python

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
    ${PYTHON_EXE} scripts/rlbench/download_peract2.py --root ${RAW_ROOT} --tasks ${TASK}

    # 2. Generate Zarr
    ZARR_OUT=${ZARR_ROOT}/${TASK}
    echo "[STEP 2] Generating Zarr in: ${ZARR_OUT}"
    
    # Clean up existing Zarr for this specific task
    rm -rf ${ZARR_OUT}/train.zarr ${ZARR_OUT}/val.zarr
    mkdir -p ${ZARR_OUT}

    ${PYTHON_EXE} data_processing/peract2_to_zarr.py \
      --root ${RAW_ROOT} \
      --tgt ${ZARR_OUT} \
      --tasks ${TASK}

    echo "Done with ${TASK}."
done

echo "------------------------------------------------------------------"
echo "All tasks processed. Zarr files are in subfolders of ${ZARR_ROOT}"
