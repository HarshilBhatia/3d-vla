#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# collect_open_drawer_3cam.sbatch
#
# Array job: 3 elements — one per camera group (G1, G2, G4).
# Collects 40 episodes of open_drawer per group, cycling through all
# task variations sequentially (variation = ep_idx % n_variations).
#
# Layout (index → group):
#   0  open_drawer / G1
#   1  open_drawer / G2
#   2  open_drawer / G4
#
# Usage:
#   sbatch scripts/orbital_cameras/collect_open_drawer_3cam.sbatch
#
# Smoke-test a single index (e.g. G1):
#   sbatch --array=0 scripts/orbital_cameras/collect_open_drawer_3cam.sbatch
#
# Resubmit a failed index (e.g. G4):
#   sbatch --array=2 scripts/orbital_cameras/collect_open_drawer_3cam.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=collect_open_drawer_3cam
#SBATCH --partition=all
#SBATCH --exclude=grogu-0-29,grogu-3-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/collect_open_drawer_3cam_%A_%a.out
#SBATCH --error=logs/collect_open_drawer_3cam_%A_%a.err

set -euo pipefail
ulimit -c 0

# ── (task, group) lookup ──────────────────────────────────────────────────────
TASKS=(   open_drawer  open_drawer  open_drawer )
MY_GROUPS=( G1           G2           G4 )

IDX="${SLURM_ARRAY_TASK_ID}"
TASK="${TASKS[$IDX]}"
GROUP="${MY_GROUPS[$IDX]}"
N_EPISODES=40

REPO_DIR="/home/harshilb/3d_flowmatch_actor"
CONTAINER="/home/harshilb/my_eval_env"
CAMERAS_FILE="${REPO_DIR}/instructions/orbital_cameras_grouped.json"
SAVE_PATH="/grogu/user/harshilb/    "

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
