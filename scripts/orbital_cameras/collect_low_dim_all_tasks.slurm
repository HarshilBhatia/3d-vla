#!/bin/bash
# Collect low_dim_obs.pkl (no images) for all PerAct tasks, 20 episodes each.
# Array index 0-16 maps to one task per job.
#
# Usage:
#   sbatch scripts/orbital_cameras/collect_low_dim_all_tasks.slurm

#SBATCH --job-name=collect_low_dim_all
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-16
#SBATCH --output=logs/collect_low_dim_all_%A_%a.out
#SBATCH --error=logs/collect_low_dim_all_%A_%a.err
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,14],grogu-3-5,grogu-3-25

set -euo pipefail
ulimit -c 0

TASKS=(
    close_jar
    insert_onto_square_peg
    light_bulb_in
    meat_off_grill
    open_drawer
    place_cups
    place_shape_in_shape_sorter
    place_wine_at_rack_location
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

TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
N_EPISODES=100
EP_START="${EP_START:-0}"
SAVE_PATH="/grogu/user/harshilb/low_dim_demos"

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
