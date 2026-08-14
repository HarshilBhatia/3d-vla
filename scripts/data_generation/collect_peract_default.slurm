#!/bin/bash
# Collect 100 PerAct demos each for light_bulb_in, stack_blocks, put_money_in_safe.
# Default PerAct cameras (left_shoulder, right_shoulder, wrist, front) — no orbital variation.
# Array: 12 jobs (4 per task × 3 tasks), each collecting 25 episodes.
#   Array index → (task, ep_start):
#     0-3  → light_bulb_in,     ep_start 0 / 25 / 50 / 75
#     4-7  → stack_blocks,      ep_start 0 / 25 / 50 / 75
#     8-11 → put_money_in_safe, ep_start 0 / 25 / 50 / 75
#
# Usage:
#   sbatch scripts/data_generation/collect_peract_default.slurm
#
# Smoke-test (single job, light_bulb_in ep 0-24):
#   sbatch --array=0 scripts/data_generation/collect_peract_default.slurm

#SBATCH --job-name=peract_collect
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/peract_collect_%A_%a.out
#SBATCH --error=logs/peract_collect_%A_%a.err
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,14],grogu-3-5,grogu-3-25

set -euo pipefail
ulimit -c 0

TASKS=(light_bulb_in stack_blocks put_money_in_safe)
N_EPISODES=25
JOBS_PER_TASK=4   # 4 × 25 = 100 episodes per task

TASK_IDX=$(( SLURM_ARRAY_TASK_ID / JOBS_PER_TASK ))
CHUNK_IDX=$(( SLURM_ARRAY_TASK_ID % JOBS_PER_TASK ))
TASK="${TASKS[$TASK_IDX]}"
EP_START=$(( CHUNK_IDX * N_EPISODES ))

SAVE_PATH="/grogu/datasets/hbhatia/peract_rollouts_new"

HOST_REPO=/home/harshilb/3d_flowmatch_actor
CONTAINER_REPO=/root/3d_flowmatch_actor
CONTAINER=/home/harshilb/my_eval_env
COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04

mkdir -p "${HOST_REPO}/logs" "${SAVE_PATH}"

echo "========================================"
echo "  Array job ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "  Node      : $(hostname)"
echo "  Started   : $(date)"
echo "  Task      : ${TASK}"
echo "  Episodes  : ${EP_START} – $(( EP_START + N_EPISODES - 1 ))"
echo "  Save      : ${SAVE_PATH}/${TASK}/"
echo "========================================"

apptainer exec \
    --nv --fakeroot --writable \
    --bind /usr/bin/xvfb-run:/usr/bin/xvfb-run \
    --bind /grogu/datasets/hbhatia:/grogu/datasets/hbhatia \
    --bind ${HOST_REPO}:${CONTAINER_REPO} \
    --env "PATH=/root/miniconda3/envs/3dfa/bin:$PATH" \
    --env "PYTHONPATH=${CONTAINER_REPO}:${PYTHONPATH:-}" \
    --env "COPPELIASIM_ROOT=${COPPELIASIM_ROOT}" \
    --env "LD_LIBRARY_PATH=${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}" \
    --env "QT_QPA_PLATFORM_PLUGIN_PATH=${COPPELIASIM_ROOT}" \
    --env "XDG_RUNTIME_DIR=/run/user/27491" \
    $CONTAINER \
    xvfb-run -a --server-args="-screen 0 1280x1024x24 +extension GLX +render" \
    python ${CONTAINER_REPO}/data/generation/generate.py \
        --save_path     "${SAVE_PATH}" \
        --tasks         "${TASK}" \
        --episodes_per_task "${N_EPISODES}" \
        --episode_start "${EP_START}" \
        --variations    -1 \
        --processes     1 \
        --seed          "${SLURM_ARRAY_TASK_ID}"

echo "========================================"
echo "  Finished  : $(date)"
echo "  Output    : ${SAVE_PATH}/${TASK}/variation*/episodes/"
echo "========================================"
