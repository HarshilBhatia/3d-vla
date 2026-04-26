#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# collect_orbital_slurm.sbatch
#
# Array job: 12 tasks (4 rlbench tasks × 3 camera groups each),
# drawn from instructions/task_group_mapping_subset.json.
# Each array element collects 30 episodes for one (task, group) pair and
# saves raw episode data to data/orbital_rollouts/{task}/{group}/episode_N/.
#
# Task/group layout (index → task / group):
#   0  insert_onto_square_peg / G1      6  push_buttons / G3
#   1  insert_onto_square_peg / G2      7  push_buttons / G4
#   2  insert_onto_square_peg / G3      8  push_buttons / G1
#   3  light_bulb_in          / G2      9  open_drawer  / G4
#   4  light_bulb_in          / G3     10  open_drawer  / G1
#   5  light_bulb_in          / G4     11  open_drawer  / G2
#
# Usage:
#   sbatch scripts/orbital_cameras/collect_orbital_slurm.sbatch
#
# Smoke-test a single index:
#   sbatch --array=0 scripts/orbital_cameras/collect_orbital_slurm.sbatch
#
# Resubmit failed indices (e.g. 3, 9):
#   sbatch --array=3,9 scripts/orbital_cameras/collect_orbital_slurm.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=orbital_collect_subset
#SBATCH --partition=all
#SBATCH --exclude=grogu-0-29
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=6-8
#SBATCH --output=logs/collect_orbital_%A_%a.out
#SBATCH --error=logs/collect_orbital_%A_%a.err

set -euo pipefail
ulimit -c 0

# ── (task, group) lookup table ────────────────────────────────────────────────
# Flattened from task_group_mapping_subset.json in insertion order.
# 4 tasks × 3 groups = 12 entries (indices 0–11).
TASKS=(
    insert_onto_square_peg  insert_onto_square_peg  insert_onto_square_peg
    light_bulb_in           light_bulb_in           light_bulb_in
    place_cups            place_cups            place_cups
    open_drawer             open_drawer             open_drawer
)

MY_GROUPS=(
    G1 G2 G3
    G2 G3 G4
    G3 G4 G1
    G4 G1 G2
)

IDX="${SLURM_ARRAY_TASK_ID}"
TASK="${TASKS[$IDX]}"
GROUP="${MY_GROUPS[$IDX]}"
N_EPISODES=30

REPO_DIR="/home/harshilb/3d_flowmatch_actor"
CONTAINER="/home/harshilb/my_eval_env"
CAMERAS_FILE="${REPO_DIR}/instructions/orbital_cameras_grouped.json"
SAVE_PATH="/grogu/user/harshilb/orbital_rollouts"

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
    --env "COLLECT_TASK=${TASK}" \
    --env "COLLECT_GROUP=${GROUP}" \
    --env "COLLECT_N_EPISODES=${N_EPISODES}" \
    --env "COLLECT_SAVE_PATH=${SAVE_PATH}" \
    --env "COLLECT_CAMERAS_FILE=${CAMERAS_FILE}" \
    --env "PYTHONPATH=/home/harshilb/3d_flowmatch_actor:${PYTHONPATH:-}" \
    "${CONTAINER}" \
    bash /home/harshilb/3d_flowmatch_actor/scripts/orbital_cameras/run_collect_inside.sh

echo "========================================"
echo "  Finished: $(date)"
echo "  Output  : ${SAVE_PATH}/${TASK}/${GROUP}/"
echo "========================================"
