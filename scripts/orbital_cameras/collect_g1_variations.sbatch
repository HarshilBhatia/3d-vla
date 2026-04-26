#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# collect_g1_variations.sbatch
#
# Array job: 4 nodes × 30 episodes = 120 episodes total for group G1.
# Each node cycles through all task variations sequentially (generate.py style)
# and writes to non-overlapping episode indices:
#   node 0 → episode_0  … episode_29   (variations 0,1,2,… mod n_var)
#   node 1 → episode_30 … episode_59
#   node 2 → episode_60 … episode_89
#   node 3 → episode_90 … episode_119
#
# Output: data/orbital_rollouts/{task}/G1/episode_{N}/
#         + episode_{N}/variation.txt recording which variation was used.
#
# Usage:
#   sbatch --export=TASK=close_jar \
#          scripts/orbital_cameras/collect_g1_variations.sbatch
#
# Single node (e.g. node 2):
#   sbatch --array=2 --export=TASK=close_jar \
#          scripts/orbital_cameras/collect_g1_variations.sbatch
#
# Override episode count per node:
#   sbatch --export=TASK=close_jar,N_EPISODES=10 \
#          scripts/orbital_cameras/collect_g1_variations.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=collect_g1_var
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/collect_g1_var_%A_%a.out
#SBATCH --error=logs/collect_g1_var_%A_%a.err
#SBATCH --exclude=grogu-3-5,grogu-0-24,grogu-0-29

set -euo pipefail
ulimit -c 0

TASK="${TASK:-open_drawer}"
GROUP="G1"
N_EPISODES="${N_EPISODES:-20}"
EP_START=$(( SLURM_ARRAY_TASK_ID * N_EPISODES ))

REPO_DIR="/home/harshilb/3d_flowmatch_actor"
CONTAINER="/home/harshilb/my_eval_env"
CAMERAS_FILE="${REPO_DIR}/instructions/orbital_cameras_grouped.json"
SAVE_PATH="/grogu/user/harshilb/orbital_rollouts_mini"

echo "========================================"
echo "  Array job ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "  Node      : $(hostname)"
echo "  Started   : $(date)"
echo "  Task      : ${TASK}  Group: ${GROUP}"
echo "  Episodes  : ${EP_START} – $(( EP_START + N_EPISODES - 1 ))"
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
    --env "COLLECT_EP_START=${EP_START}" \
    --env "COLLECT_SAVE_PATH=${SAVE_PATH}" \
    --env "COLLECT_CAMERAS_FILE=${CAMERAS_FILE}" \
    --env "PYTHONPATH=/home/harshilb/3d_flowmatch_actor:${PYTHONPATH:-}" \
    "${CONTAINER}" \
    bash /home/harshilb/3d_flowmatch_actor/scripts/orbital_cameras/run_collect_g1_variations_inside.sh

echo "========================================"
echo "  Finished: $(date)"
echo "  Output  : ${SAVE_PATH}/${TASK}/${GROUP}/"
echo "========================================"
