#!/bin/bash
# Online eval: default_3dfa × 3 tasks (task-specific camera groups) × no noise.
#
#   idx 0 -> light_bulb_in  G4
#   idx 1 -> place_cups     G1
#   idx 2 -> stack_cups     G4
#
# Usage:
#   sbatch scripts/eval/online_eval_default_taskcam_nonoise.slurm
#SBATCH --job-name=eval_default_taskcam_nonoise
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/eval_default_taskcam_nonoise_%A_%a.out
#SBATCH --error=logs/eval_default_taskcam_nonoise_%A_%a.err
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,14],grogu-2-22,grogu-3-5,grogu-3-25

set -euo pipefail
ulimit -c 0

echo "Job started at: $(date)"
echo "Node: $(hostname)  Array index: $SLURM_ARRAY_TASK_ID"

HOST_REPO=/home/harshilb/3d_flowmatch_actor
CONTAINER_REPO=/root/3d_flowmatch_actor
CONTAINER=/home/harshilb/my_eval_env
COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04

TASKS=(light_bulb_in place_cups stack_cups)
CAM_GROUPS=(G4 G1 G4)

TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
GGROUP="${CAM_GROUPS[$SLURM_ARRAY_TASK_ID]}"

CHECKPOINT="${CONTAINER_REPO}/train_logs/full_dataset/default_3dfa/last.pth"
OUTPUT_FILE="${HOST_REPO}/eval_logs/24may_default_taskcam_nonoise/${GGROUP}/results_${TASK}.json"

mkdir -p "$(dirname "$OUTPUT_FILE")" logs

echo "Task:       $TASK"
echo "GGroup:     $GGROUP"
echo "Checkpoint: $CHECKPOINT"
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
