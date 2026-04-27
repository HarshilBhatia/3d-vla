#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# eval_orbital_apptainer.slurm
#
# Slurm array job: evaluate all 18 PerAct orbital tasks in parallel inside
# the apptainer container on grogu (one job per task).
#
# Usage:
#   sbatch --export=CHECKPOINT=/path/to/checkpoint.pth,RUN_LOG_DIR=my_run \
#          scripts/eval/eval_orbital_apptainer.slurm
#
# Single task (index 5 = open_drawer):
#   sbatch --array=5 --export=CHECKPOINT=...,RUN_LOG_DIR=... \
#          scripts/eval/eval_orbital_apptainer.slurm
#
# Extra Hydra overrides (e.g. miscalibration_noise_level=medium):
#   sbatch --export=CHECKPOINT=...,RUN_LOG_DIR=...,EXTRA="miscalibration_noise_level=medium" ...
#
# Task index mapping (0-based):
#   0  place_cups                    9  put_groceries_in_cupboard
#   1  close_jar                    10  put_item_in_drawer
#   2  insert_onto_square_peg       11  put_money_in_safe
#   3  light_bulb_in                12  reach_and_drag
#   4  meat_off_grill               13  slide_block_to_color_target
#   5  open_drawer                  14  stack_blocks
#   6  place_shape_in_shape_sorter  15  stack_cups
#   7  place_wine_at_rack_location  16  sweep_to_dustpan_of_size
#   8  push_buttons                 17  turn_tap
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=eval_orbital
#SBATCH --partition=shubhamlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/eval_orbital_%A_%a.out
#SBATCH --error=logs/eval_orbital_%A_%a.err

set -euo pipefail
ulimit -c 0

echo "Job started at: $(date)"
echo "Node: $(hostname)  Array index: $SLURM_ARRAY_TASK_ID"

HOST_REPO=/home/harshilb/3d_flowmatch_actor
CONTAINER_REPO=/root/3d_flowmatch_actor
CONTAINER=/home/harshilb/my_eval_env
COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04

# TASKS=(
#     place_cups close_jar insert_onto_square_peg
#     light_bulb_in meat_off_grill open_drawer
#     place_shape_in_shape_sorter place_wine_at_rack_location
#     push_buttons put_groceries_in_cupboard
#     put_item_in_drawer put_money_in_safe reach_and_drag
#     slide_block_to_color_target stack_blocks stack_cups
#     sweep_to_dustpan_of_size turn_tap
# )
TASKS=(open_drawer)

TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"

# : "${CHECKPOINT:?'CHECKPOINT not set — pass via --export=CHECKPOINT=...'}"
# : "${RUN_LOG_DIR:?'RUN_LOG_DIR not set — pass via --export=RUN_LOG_DIR=...'}"
RUN_LOG_DIR='open_drawer_test'
CHECKPOINT='/root/3d_flowmatch_actor/train_logs/Orbital/open_drawer_test/best.pth'

EXTRA="${EXTRA:-}"

OUTPUT_DIR="${HOST_REPO}/eval_logs/Orbital/${RUN_LOG_DIR}"
OUTPUT_FILE="${CONTAINER_REPO}/eval_logs/Orbital/${RUN_LOG_DIR}/results_${TASK}.json"

mkdir -p "$OUTPUT_DIR" logs

echo "Task:       $TASK"
echo "Checkpoint: $CHECKPOINT"
echo "Output:     ${OUTPUT_DIR}/results_${TASK}.json"

if [[ -f "${OUTPUT_DIR}/results_${TASK}.json" ]]; then
    echo "[SKIP] Results already exist."
    exit 0
fi

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
    bash ${CONTAINER_REPO}/scripts/eval/eval_orbital.sh \
        checkpoint="$CHECKPOINT" \
        output_file="$OUTPUT_FILE" \
        task="$TASK" \
        $EXTRA

echo "Done at: $(date)"
