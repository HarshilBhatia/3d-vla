#!/bin/bash
# Checkpoint sweep: default_3dfa (all interm + last) × 2 tasks, train-group cameras, GT demos.
# data_dir = low_dim_demos (GT resets), spawn_camera_group = training group, camera_groups=GT.
#
# Tasks:
#   slide_block_to_color_target -> spawn G2
#   reach_and_drag              -> spawn G1
#
# Checkpoints (ckpt_idx):
#   0: interm_step_100000
#   1: interm_step_150000
#   2: interm_step_200000
#   3: interm_step_250000
#   4: interm_step_300000
#   5: interm_step_350000
#   6: interm_step_400000
#   7: last
#
# Array layout: ckpt_idx = idx / 2, task_idx = idx % 2
#   idx 0  -> 100k  / slide_block G2
#   idx 1  -> 100k  / reach_and_drag G1
#   idx 2  -> 150k  / slide_block G2
#   idx 3  -> 150k  / reach_and_drag G1
#   ...
#   idx 14 -> last  / slide_block G2
#   idx 15 -> last  / reach_and_drag G1
#
# Usage:
#   sbatch scripts/eval/online_eval_default_ckpt_sweep.slurm
#SBATCH --job-name=eval_default_ckpt_sweep
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/eval_default_ckpt_sweep_%A_%a.out
#SBATCH --error=logs/eval_default_ckpt_sweep_%A_%a.err
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,14],grogu-2-22,grogu-3-5,grogu-3-25

set -euo pipefail
ulimit -c 0

echo "Job started at: $(date)"
echo "Node: $(hostname)  Array index: $SLURM_ARRAY_TASK_ID"

HOST_REPO=/home/harshilb/3d_flowmatch_actor
CONTAINER_REPO=/root/3d_flowmatch_actor
CONTAINER=/home/harshilb/my_eval_env
COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04

CKPT_NAMES=(interm_step_100000 interm_step_150000 interm_step_200000 interm_step_250000 interm_step_300000 interm_step_350000 interm_step_400000 last)
CKPT_FILES=(interm_step_100000.pth interm_step_150000.pth interm_step_200000.pth interm_step_250000.pth interm_step_300000.pth interm_step_350000.pth interm_step_400000.pth last.pth)

TASKS=(slide_block_to_color_target reach_and_drag)
CAM_GROUPS=(G2 G1)

CKPT_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
TASK_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))

CKPT_NAME="${CKPT_NAMES[$CKPT_IDX]}"
CKPT_FILE="${CKPT_FILES[$CKPT_IDX]}"
TASK="${TASKS[$TASK_IDX]}"
GGROUP="${CAM_GROUPS[$TASK_IDX]}"

CHECKPOINT="${CONTAINER_REPO}/train_logs/full_dataset/default_3dfa/${CKPT_FILE}"
OUTPUT_FILE="${HOST_REPO}/eval_logs/24may_default_ckpt_sweep/${CKPT_NAME}/${GGROUP}/results_${TASK}.json"

mkdir -p "$(dirname "$OUTPUT_FILE")" logs

echo "Checkpoint: $CKPT_NAME"
echo "Task:       $TASK"
echo "GGroup:     $GGROUP"
echo "Output:     $OUTPUT_FILE"

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
    python ${CONTAINER_REPO}/online_evaluation_rlbench/evaluate_policy.py \
        dataset=OrbitalWrist \
        data_dir=/grogu/user/harshilb/low_dim_demos \
        cameras_file=instructions/orbital_cameras_grouped.json \
        task_group_mapping_file=instructions/task_group_mapping.json \
        val_instructions=instructions/peract/instructions.json \
        "image_size='256,256'" \
        bimanual=false \
        checkpoint="$CHECKPOINT" \
        output_file="$OUTPUT_FILE" \
        task="$TASK" \
        camera_groups=GT \
        spawn_camera_group="$GGROUP" \
        num_demos=20 \
        save_video=true \
        "$@"

echo "Done at: $(date)"
