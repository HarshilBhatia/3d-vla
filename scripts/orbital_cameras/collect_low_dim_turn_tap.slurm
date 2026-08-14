#!/bin/bash
# Collect low_dim_obs.pkl only (no images) for turn_tap, 20 episodes.
#
# Usage:
#   sbatch scripts/orbital_cameras/collect_low_dim_turn_tap.slurm

#SBATCH --job-name=collect_low_dim_turn_tap
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/collect_low_dim_turn_tap_%j.out
#SBATCH --error=logs/collect_low_dim_turn_tap_%j.err
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,14],grogu-3-5,grogu-3-25

set -euo pipefail
ulimit -c 0

TASK="turn_tap"
N_EPISODES=20
EP_START="${EP_START:-0}"
SAVE_PATH="/grogu/user/harshilb/low_dim_demos"

HOST_REPO=/home/harshilb/3d_flowmatch_actor
CONTAINER_REPO=/root/3d_flowmatch_actor
CONTAINER=/home/harshilb/my_eval_env
COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04

mkdir -p "${HOST_REPO}/logs" "${SAVE_PATH}"

echo "========================================"
echo "  Job      : ${SLURM_JOB_ID}"
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
