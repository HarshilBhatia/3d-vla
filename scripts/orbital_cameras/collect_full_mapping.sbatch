#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Collect all 54 (task, group) pairs from instructions/task_group_mapping.json.
# 18 tasks × 3 groups = 54 array indices (0–53), 40 episodes each.
#
# Usage:
#   sbatch scripts/orbital_cameras/collect_full_mapping.sbatch
#
# Smoke-test a single index:
#   sbatch --array=0 scripts/orbital_cameras/collect_full_mapping.sbatch
#
# Resubmit failed indices (e.g. 3, 7):
#   sbatch --array=3,7 scripts/orbital_cameras/collect_full_mapping.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=collect_full_mapping
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-53
#SBATCH --output=logs/collect_full_%A_%a.out
#SBATCH --error=logs/collect_full_%A_%a.err
#SBATCH --exclude=grogu-0-29,grogu-3-5,grogu-0-14

set -euo pipefail
ulimit -c 0

# ── (task, group) lookup table ────────────────────────────────────────────────
# Flattened from instructions/task_group_mapping.json in insertion order.
# 18 tasks × 3 groups = 54 entries (indices 0–53).
TASKS=(
    place_cups                  place_cups                  place_cups
    close_jar                   close_jar                   close_jar
    insert_onto_square_peg      insert_onto_square_peg      insert_onto_square_peg
    light_bulb_in               light_bulb_in               light_bulb_in
    meat_off_grill              meat_off_grill              meat_off_grill
    open_drawer                 open_drawer                 open_drawer
    place_shape_in_shape_sorter place_shape_in_shape_sorter place_shape_in_shape_sorter
    place_wine_at_rack_location place_wine_at_rack_location place_wine_at_rack_location
    push_buttons                push_buttons                push_buttons
    put_groceries_in_cupboard   put_groceries_in_cupboard   put_groceries_in_cupboard
    put_item_in_drawer          put_item_in_drawer          put_item_in_drawer
    put_money_in_safe           put_money_in_safe           put_money_in_safe
    reach_and_drag              reach_and_drag              reach_and_drag
    slide_block_to_color_target slide_block_to_color_target slide_block_to_color_target
    stack_blocks                stack_blocks                stack_blocks
    stack_cups                  stack_cups                  stack_cups
    sweep_to_dustpan_of_size    sweep_to_dustpan_of_size    sweep_to_dustpan_of_size
    turn_tap                    turn_tap                    turn_tap
)

CAM_GROUPS=(
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
GROUP="${CAM_GROUPS[$IDX]}"
N_EPISODES=40

REPO_DIR="/home/harshilb/3d_flowmatch_actor"
CONTAINER="/home/harshilb/my_eval_env"
CAMERAS_FILE="${REPO_DIR}/instructions/orbital_cameras_grouped.json"
SAVE_PATH="/grogu/datasets/hbhatia/full_rollouts"
CHECK_PATH="/grogu/user/harshilb/full_rollouts"

echo "========================================"
echo "  Array job ${SLURM_ARRAY_JOB_ID}[${IDX}]"
echo "  Node      : $(hostname)"
echo "  Started   : $(date)"
echo "  Task      : ${TASK}  Group: ${GROUP}"
echo "  Episodes  : 0 – $(( N_EPISODES - 1 ))"
echo "  Save      : ${SAVE_PATH}/${TASK}/${GROUP}/"
echo "========================================"

mkdir -p "${REPO_DIR}/logs" "${SAVE_PATH}"

unset DISPLAY
unset QT_QPA_PLATFORM

apptainer exec \
    --nv \
    --fakeroot \
    --bind /usr/bin/xvfb-run:/usr/bin/xvfb-run \
    --bind /home/harshilb/:/home/harshilb/ \
    --bind /grogu/user/harshilb/:/grogu/user/harshilb/ \
    --bind /grogu/datasets/hbhatia/:/grogu/datasets/hbhatia/ \
    --env "COLLECT_TASK=${TASK}" \
    --env "COLLECT_GROUP=${GROUP}" \
    --env "COLLECT_N_EPISODES=${N_EPISODES}" \
    --env "COLLECT_SAVE_PATH=${SAVE_PATH}" \
    --env "COLLECT_CHECK_PATH=${CHECK_PATH}" \
    --env "COLLECT_CAMERAS_FILE=${CAMERAS_FILE}" \
    --env "PYTHONPATH=/home/harshilb/3d_flowmatch_actor:${PYTHONPATH:-}" \
    "${CONTAINER}" \
    bash /home/harshilb/3d_flowmatch_actor/scripts/orbital_cameras/run_collect_inside.sh

echo "========================================"
echo "  Finished: $(date)"
echo "  Output  : ${SAVE_PATH}/${TASK}/${GROUP}/"
echo "========================================"
