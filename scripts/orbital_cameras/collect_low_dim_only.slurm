#!/bin/bash
# Collect low_dim_obs.pkl only (no images) for one or more RLBench tasks.
# Each array index maps to one task; all episodes collected in one session.
#
# Usage:
#   sbatch scripts/orbital_cameras/collect_low_dim_only.slurm
#   sbatch --array=0 ...   # single task smoke-test
#
# For parallel episode collection (e.g. two nodes, 50 eps each):
#   EP_START=0   sbatch scripts/orbital_cameras/collect_low_dim_only.slurm
#   EP_START=50  sbatch scripts/orbital_cameras/collect_low_dim_only.slurm

#SBATCH --job-name=collect_low_dim
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/collect_low_dim_%A_%a.out
#SBATCH --error=logs/collect_low_dim_%A_%a.err

set -euo pipefail
ulimit -c 0

# ── Task list (one per array index) ──────────────────────────────────────────
TASKS=(
    open_drawer
)

TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
N_EPISODES=90
EP_START="${EP_START:-0}"
SAVE_PATH="/grogu/user/harshilb/low_dim_demos"

# ── Paths ─────────────────────────────────────────────────────────────────────
HOST_REPO=/home/harshilb/3d_flowmatch_actor
CONTAINER_REPO=/root/3d_flowmatch_actor
CONTAINER=/home/harshilb/my_eval_env
COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04

mkdir -p "${HOST_REPO}/logs" "${SAVE_PATH}"

echo "========================================"
echo "  Array job ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "  Node     : $(hostname)"
echo "  Started  : $(date)"
echo "  Task     : ${TASK}"
echo "  Episodes : ${EP_START} – $(( EP_START + N_EPISODES - 1 ))"
echo "  Save     : ${SAVE_PATH}/${TASK}/GT/"
echo "========================================"

apptainer exec \
    --nv --fakeroot --writable \
    --pwd ${CONTAINER_REPO} \
    --bind /usr/bin/xvfb-run:/usr/bin/xvfb-run \
    --bind /grogu/user/harshilb:/grogu/user/harshilb \
    --bind ${HOST_REPO}:${CONTAINER_REPO} \
    --env "PATH=/root/miniconda3/envs/3dfa/bin:$PATH" \
    --env "PYTHONPATH=${CONTAINER_REPO}:${PYTHONPATH:-}" \
    --env "COPPELIASIM_ROOT=${COPPELIASIM_ROOT}" \
    --env "LD_LIBRARY_PATH=${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}" \
    --env "QT_QPA_PLATFORM_PLUGIN_PATH=${COPPELIASIM_ROOT}" \
    --env "XDG_RUNTIME_DIR=/run/user/27491" \
    $CONTAINER \
    xvfb-run -a --server-args="-screen 0 1280x1024x24 +extension GLX +render" \
    python scripts/orbital_cameras/collect_low_dim_only.py \
        --tasks "$TASK" \
        --n-episodes "$N_EPISODES" \
        --ep-start "$EP_START" \
        --save-path "$SAVE_PATH"

echo "========================================"
echo "  Finished : $(date)"
echo "  Output   : ${SAVE_PATH}/${TASK}/GT/"
echo "========================================"
