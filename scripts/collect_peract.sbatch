#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# collect_peract.sbatch
#
# Array job: up to 4 nodes, each handling a non-overlapping slice of
# variations. Collects N_EPISODES episodes per variation.
#
# Variation assignment (default VARS_PER_NODE=30):
#   node 0 → variations  0 –  29
#   node 1 → variations 30 –  59
#   node 2 → variations 60 –  89
#   node 3 → variations 90 – 119
#
# Nodes whose offset exceeds the task's actual variation count exit cleanly.
#
# Output: {SAVE_PATH}/{TASK}/variation{N}/episodes/episode{M}/
#
# Usage (120 total episodes = 4 nodes × 30 variations × 1 episode/var):
#   sbatch --export=TASK=push_buttons \
#          scripts/collect_peract.sbatch
#
# For tasks with few variations (e.g. open_drawer has 1):
# nodes 1-3 will exit immediately; only node 0 collects.
# Set N_EPISODES higher to compensate:
#   sbatch --export=TASK=open_drawer,N_EPISODES=120 \
#          scripts/collect_peract.sbatch
#
# Override nodes / chunk size:
#   sbatch --array=0-1 --export=TASK=close_jar,VARS_PER_NODE=15,N_EPISODES=4 \
#          scripts/collect_peract.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=collect_peract
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/collect_peract_%A_%a.out
#SBATCH --error=logs/collect_peract_%A_%a.err
#SBATCH --exclude=grogu-3-5

set -euo pipefail
ulimit -c 0

TASK="${TASK:-open_drawer}"
N_EPISODES="${N_EPISODES:-1}"
VARS_PER_NODE="${VARS_PER_NODE:-20}"
VAR_OFFSET=$(( SLURM_ARRAY_TASK_ID * VARS_PER_NODE ))

REPO_DIR="/home/harshilb/3d_flowmatch_actor"
CONTAINER="/home/harshilb/my_eval_env"
SAVE_PATH="/grogu/user/harshilb/peract_rollouts"

echo "========================================"
echo "  Array job ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "  Node      : $(hostname)"
echo "  Started   : $(date)"
echo "  Task      : ${TASK}"
echo "  Variations: ${VAR_OFFSET} – $(( VAR_OFFSET + VARS_PER_NODE - 1 ))"
echo "  Eps/var   : ${N_EPISODES}"
echo "  Save      : ${SAVE_PATH}/${TASK}/"
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
    --env "COLLECT_SAVE_PATH=${SAVE_PATH}" \
    --env "COLLECT_N_EPISODES=${N_EPISODES}" \
    --env "COLLECT_VAR_OFFSET=${VAR_OFFSET}" \
    --env "COLLECT_VARS_PER_NODE=${VARS_PER_NODE}" \
    --env "PYTHONPATH=/home/harshilb/3d_flowmatch_actor:${PYTHONPATH:-}" \
    "${CONTAINER}" \
    bash /home/harshilb/3d_flowmatch_actor/scripts/run_collect_peract_inside.sh

echo "========================================"
echo "  Finished: $(date)"
echo "  Output  : ${SAVE_PATH}/${TASK}/"
echo "========================================"
