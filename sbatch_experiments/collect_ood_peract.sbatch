#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# collect_ood_peract.sbatch
#
# Slurm array job: one element per RLBench task.
# Collects --episodes_per_task demos at variation 0 using OOD cameras
# (ood_camera.json: ood_az30_el40 + ood_az330_el55 + wrist).
#
# Output format (standard RLBench / get_stored_demos compatible):
#   data/peract_raw_ood/{task}/variation0/episodes/episode{N}/
#       orbital_left_rgb/   orbital_left_depth/
#       orbital_right_rgb/  orbital_right_depth/
#       wrist_rgb/          wrist_depth/
#       low_dim_obs.pkl     variation_number.pkl
#       ood_extrinsics.pkl
#
# Usage:
#   cd /ocean/projects/cis240058p/hbhatia1/3d-vla
#   sbatch sbatch_experiments/collect_ood_peract.sbatch
#
# Smoke-test a single task (index 0 = close_jar):
#   sbatch --array=0 sbatch_experiments/collect_ood_peract.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=ood_peract_collect
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=08:00:00
#SBATCH --array=0-18
#SBATCH --output=logs/ood_peract_%A_%a.out
#SBATCH --error=logs/ood_peract_%A_%a.err

set -euo pipefail
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND (exit $?)" >&2' ERR
ulimit -c 0

# ── Config ────────────────────────────────────────────────────────────────────
REPO_DIR="/ocean/projects/cis240058p/hbhatia1/3d-vla"
CONTAINER="/ocean/projects/cis240058p/hbhatia1/containers/3dfa-sandbox.sif"
COPPELIASIM_ROOT="${REPO_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"

# SAVE_PATH="${REPO_DIR}/data/peract_raw_ood"
SAVE_PATH="${REPO_DIR}/data/peract_G1_data"
# CAMERA_FILE="${REPO_DIR}/ood_camera.json"
CAMERA_FILE="${REPO_DIR}/orbital_cameras_grouped.json"
EPISODES="${EPISODES:-100}"
VARIATIONS="${VARIATIONS:-1}"       # 1 = variation 0 only
IMAGE_SIZE="${IMAGE_SIZE:-128}"
FOV_DEG="${FOV_DEG:-60.0}"

# ── Task lookup (one per array index) ────────────────────────────────────────
TASKS=(
    place_cups
    close_jar
    insert_onto_square_peg
    light_bulb_in
    meat_off_grill
    open_drawer
    place_shape_in_shape_sorter
    place_wine_at_rack_location
    push_buttons
    put_groceries_in_cupboard
    put_item_in_drawer
    put_money_in_safe
    reach_and_drag
    slide_block_to_color_target
    stack_blocks
    stack_cups
    sweep_to_dustpan_of_size
    turn_tap
)

IDX="${SLURM_ARRAY_TASK_ID}"
TASK="${TASKS[$IDX]}"

# ── Info ──────────────────────────────────────────────────────────────────────
echo "========================================"
echo "  Array job ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node   : $(hostname)"
echo "  Started: $(date)"
echo "  Task   : ${TASK}"
echo "  Episodes: ${EPISODES}  Variations: ${VARIATIONS}"
echo "  Save   : ${SAVE_PATH}/${TASK}/"
echo "========================================"

mkdir -p "${REPO_DIR}/logs" "${SAVE_PATH}"

unset DISPLAY
unset QT_QPA_PLATFORM


#xvfb-run -a python RLBench/tools/dataset_generator.py \
#       --save_path data/peract_raw_g1 \
#       --tasks close_jar \
#       --episodes_per_task 5 \
#       --all_variations True \
#       --camera_file orbital_cameras_grouped.json \
#       --camera_group G1

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
            --camera_group G1


echo "========================================"
echo "  Finished: $(date)"
echo "  Output  : ${SAVE_PATH}/${TASK}/variation0/episodes/"
echo "========================================"
