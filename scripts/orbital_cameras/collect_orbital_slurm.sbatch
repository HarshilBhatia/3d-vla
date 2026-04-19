#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# collect_orbital_slurm.sbatch
#
# Slurm array job: 54 tasks (18 rlbench tasks × 3 camera groups each).
# Each array element collects 30 episodes for one (task, group) pair and
# saves raw episode data to data/orbital_rollouts/{task}/{group}/episode_N/.
#
# Uses xvfb-run for headless X11 rendering (CPU, no GPU required).
#
# Usage:
#   cd /ocean/projects/cis240058p/hbhatia1/3d-vla
#   sbatch scripts/rlbench/collect_orbital_slurm.sbatch
#
# Smoke test a single index first:
#   sbatch --array=3 scripts/rlbench/collect_orbital_slurm.sbatch
#   # index 3 = close_jar / G2
#
# Resubmit failed indices (e.g. 7, 23, 41):
#   sbatch --array=7,23,41 scripts/rlbench/collect_orbital_slurm.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=orbital_collect
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
#SBATCH --time=06:00:00
#SBATCH --array=0-53
#SBATCH --output=logs/collect_orbital_%A_%a.out
#SBATCH --error=logs/collect_orbital_%A_%a.err

set -euo pipefail
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND (exit $?)" >&2' ERR

# Suppress CoppeliaSim crash core dumps (multi-GB files fill /tmp quickly)
ulimit -c 0

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="/ocean/projects/cis240058p/hbhatia1/3d-vla"
CONTAINER="/ocean/projects/cis240058p/hbhatia1/containers/3dfa-sandbox.sif"
SAVE_PATH="${REPO_DIR}/data/orbital_rollouts"
CAMERAS_FILE="${REPO_DIR}/orbital_cameras_grouped.json"
SCRIPT="${REPO_DIR}/scripts/orbital_cameras/collect.py"

# ── (task, group) lookup table ────────────────────────────────────────────────
# Flattened from task_group_mapping.json in insertion order.
# 18 tasks × 3 groups = 54 entries (indices 0–53).
TASKS=(
    place_cups                    place_cups                    place_cups
    close_jar                     close_jar                     close_jar
    insert_onto_square_peg        insert_onto_square_peg        insert_onto_square_peg
    light_bulb_in                 light_bulb_in                 light_bulb_in
    meat_off_grill                meat_off_grill                meat_off_grill
    open_drawer                   open_drawer                   open_drawer
    place_shape_in_shape_sorter   place_shape_in_shape_sorter   place_shape_in_shape_sorter
    place_wine_at_rack_location   place_wine_at_rack_location   place_wine_at_rack_location
    push_buttons                  push_buttons                  push_buttons
    put_groceries_in_cupboard     put_groceries_in_cupboard     put_groceries_in_cupboard
    put_item_in_drawer            put_item_in_drawer            put_item_in_drawer
    put_money_in_safe             put_money_in_safe             put_money_in_safe
    reach_and_drag                reach_and_drag                reach_and_drag
    slide_block_to_color_target   slide_block_to_color_target   slide_block_to_color_target
    stack_blocks                  stack_blocks                  stack_blocks
    stack_cups                    stack_cups                    stack_cups
    sweep_to_dustpan_of_size      sweep_to_dustpan_of_size      sweep_to_dustpan_of_size
    turn_tap                      turn_tap                      turn_tap
)

MY_GROUPS=(
    G1 G2 G3
    G2 G3 G4
    G3 G4 G5
    G4 G5 G6
    G5 G6 G1
    G6 G1 G2
    G1 G2 G3
    G2 G3 G4
    G3 G4 G5
    G4 G5 G6
    G5 G6 G1
    G6 G1 G2
    G1 G2 G3
    G2 G3 G4
    G3 G4 G5
    G4 G5 G6
    G5 G6 G1
    G6 G1 G2
)

IDX="${SLURM_ARRAY_TASK_ID}"
TASK="${TASKS[$IDX]}"
GROUP="${MY_GROUPS[$IDX]}"


# ── Info ──────────────────────────────────────────────────────────────────────
echo "========================================"
echo "  Array job ${SLURM_ARRAY_JOB_ID}[${IDX}]"
echo "  Node   : $(hostname)"
echo "  Started: $(date)"
echo "  Task   : ${TASK}  Group: ${GROUP}"
echo "  Save   : ${SAVE_PATH}/${TASK}/${GROUP}/"
echo "========================================"

mkdir -p "${REPO_DIR}/logs" "${SAVE_PATH}"

# Use X11 via xvfb; let CoppeliaSim pick its default platform (no eglfs)
unset DISPLAY
unset QT_QPA_PLATFORM

COPPELIASIM_ROOT="${REPO_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"

xvfb-run -a \
    apptainer exec \
        --env "COPPELIASIM_ROOT=${COPPELIASIM_ROOT}" \
        --env "LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}" \
        --env "QT_QPA_PLATFORM_PLUGIN_PATH=${COPPELIASIM_ROOT}" \
        --bind "${REPO_DIR}:${REPO_DIR}" \
        "${CONTAINER}" \
        python3 "${SCRIPT}" \
            --task         "${TASK}" \
            --groups       "${GROUP}" \
            --n-episodes   30 \
            --save-path    "${SAVE_PATH}" \
            --cameras-file "${CAMERAS_FILE}" \
            --image-size   256 \
            --fov-deg      60.0

echo "========================================"
echo "  Finished: $(date)"
echo "  Output  : ${SAVE_PATH}/${TASK}/${GROUP}/"
echo "========================================"
