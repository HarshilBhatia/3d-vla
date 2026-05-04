#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# collect_subset.sbatch
#
# Slurm array job: one element per (task, camera_group) pair defined in
# instructions/task_group_mapping_subset.json.
#
# Tasks x Groups  (9 total, array indices 0-8):
#   0  insert_onto_square_peg  G1
#   1  insert_onto_square_peg  G2
#   2  insert_onto_square_peg  G3
#   3  light_bulb_in           G1
#   4  light_bulb_in           G2
#   5  light_bulb_in           G3
#   6  push_buttons            G1
#   7  push_buttons            G2
#   8  push_buttons            G3
#
# Usage:
#   cd /ocean/projects/cis240058p/hbhatia1/3d-vla
#   sbatch scripts/data_generation/collect_subset.sh
#
# Smoke-test a single pair (index 0):
#   sbatch --array=0 scripts/data_generation/collect_subset.sh
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=collect_subset
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=08:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/collect_subset_%A_%a.out
#SBATCH --error=logs/collect_subset_%A_%a.err

set -euo pipefail
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND (exit $?)" >&2' ERR
ulimit -c 0

# ── Config ────────────────────────────────────────────────────────────────────
REPO_DIR="/ocean/projects/cis240058p/hbhatia1/3d-vla"
CONTAINER="/ocean/projects/cis240058p/hbhatia1/containers/3dfa-sandbox.sif"
COPPELIASIM_ROOT="${REPO_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"

CAMERA_FILE="${REPO_DIR}/instructions/orbital_cameras_grouped.json"
EPISODES="${EPISODES:-100}"
VARIATIONS="${VARIATIONS:-1}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
FOV_DEG="${FOV_DEG:-60.0}"

# ── Task × Group lookup ───────────────────────────────────────────────────────
TASKS=(
    open_drawer open_drawer open_drawer
)
GROUPS=(
    G1 G2 G3
)

IDX="${SLURM_ARRAY_TASK_ID}"
TASK="${TASKS[$IDX]}"
GROUP="${GROUPS[$IDX]}"

SAVE_PATH="${REPO_DIR}/data/peract_subset_${GROUP}"

# ── Info ──────────────────────────────────────────────────────────────────────
echo "========================================"
echo "  Array job ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node     : $(hostname)"
echo "  Started  : $(date)"
echo "  Task     : ${TASK}"
echo "  Group    : ${GROUP}"
echo "  Episodes : ${EPISODES}  Variations: ${VARIATIONS}"
echo "  Save     : ${SAVE_PATH}/${TASK}/"
echo "========================================"

mkdir -p "${REPO_DIR}/logs" "${SAVE_PATH}"

unset DISPLAY
unset QT_QPA_PLATFORM

# ── Collect ───────────────────────────────────────────────────────────────────
xvfb-run -a \
    apptainer exec \
        --env "COPPELIASIM_ROOT=${COPPELIASIM_ROOT}" \
        --env "LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}" \
        --env "QT_QPA_PLATFORM_PLUGIN_PATH=${COPPELIASIM_ROOT}" \
        --env "PYTHONPATH=${REPO_DIR}/RLBench:${REPO_DIR}" \
        --bind "${REPO_DIR}:${REPO_DIR}" \
        "${CONTAINER}" \
        python3 "${REPO_DIR}/RLBench/tools/dataset_generator.py" \
            --save_path         "${SAVE_PATH}" \
            --tasks             "${TASK}" \
            --episodes_per_task "${EPISODES}" \
            --all_variations    False \
            --variations        "${VARIATIONS}" \
            --image_size        "${IMAGE_SIZE}" "${IMAGE_SIZE}" \
            --camera_file       "${CAMERA_FILE}" \
            --fov_deg           "${FOV_DEG}" \
            --camera_group      "${GROUP}"

echo "========================================"
echo "  Finished : $(date)"
echo "  Output   : ${SAVE_PATH}/${TASK}/variation0/episodes/"
echo "========================================"
