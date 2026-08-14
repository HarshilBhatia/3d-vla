#!/bin/bash
# Online eval: default_3dfa × 5 missed/failed tasks, train set, no noise.
# 3 from full_rollouts_merged (had no data in full_rollouts):
#   open_drawer, slide_block_to_color_target, sweep_to_dustpan_of_size
# 2 resubmits from full_rollouts (crashed / hung):
#   put_item_in_drawer, place_shape_in_shape_sorter
#
#   idx 0 -> open_drawer                  G1  full_rollouts_merged (40 eps)
#   idx 1 -> slide_block_to_color_target  G2  full_rollouts_merged (40 eps)
#   idx 2 -> sweep_to_dustpan_of_size     G1  full_rollouts_merged (40 eps)
#   idx 3 -> put_item_in_drawer           G1  full_rollouts        (38 eps)
#   idx 4 -> place_shape_in_shape_sorter  G1  full_rollouts        (27 eps)
#
# Usage:
#   sbatch scripts/eval/online_eval_default_trainset_missed.slurm
#SBATCH --job-name=eval_default_trainset_missed
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/eval_default_trainset_missed_%A_%a.out
#SBATCH --error=logs/eval_default_trainset_missed_%A_%a.err
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,14],grogu-2-22,grogu-3-5,grogu-3-25

set -euo pipefail
ulimit -c 0

echo "Job started at: $(date)"
echo "Node: $(hostname)  Array index: $SLURM_ARRAY_TASK_ID"

HOST_REPO=/home/harshilb/3d_flowmatch_actor
CONTAINER_REPO=/root/3d_flowmatch_actor
CONTAINER=/home/harshilb/my_eval_env
COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04

TASKS=(open_drawer slide_block_to_color_target sweep_to_dustpan_of_size put_item_in_drawer place_shape_in_shape_sorter)
CAM_GROUPS=(G1 G2 G1 G1 G1)
DATA_DIRS=(
    /grogu/user/harshilb/full_rollouts_merged
    /grogu/user/harshilb/full_rollouts_merged
    /grogu/user/harshilb/full_rollouts_merged
    /grogu/datasets/hbhatia/full_rollouts
    /grogu/datasets/hbhatia/full_rollouts
)

TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
GGROUP="${CAM_GROUPS[$SLURM_ARRAY_TASK_ID]}"
DATA_DIR="${DATA_DIRS[$SLURM_ARRAY_TASK_ID]}"

CHECKPOINT="${CONTAINER_REPO}/train_logs/full_dataset/default_3dfa/last.pth"
OUTPUT_FILE="${HOST_REPO}/eval_logs/24may_default_trainset_missed/${GGROUP}/results_${TASK}.json"

mkdir -p "$(dirname "$OUTPUT_FILE")" logs

echo "Task:       $TASK"
echo "GGroup:     $GGROUP"
echo "Data dir:   $DATA_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "Output:     $OUTPUT_FILE"

apptainer exec \
    --nv --fakeroot --writable \
    --pwd ${CONTAINER_REPO} \
    --bind /usr/bin/xvfb-run:/usr/bin/xvfb-run \
    --bind /grogu:/grogu \
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
        data_dir="${DATA_DIR}" \
        cameras_file=instructions/orbital_cameras_grouped.json \
        task_group_mapping_file=instructions/task_group_mapping.json \
        val_instructions=instructions/peract/instructions.json \
        "image_size='256,256'" \
        bimanual=false \
        checkpoint="$CHECKPOINT" \
        output_file="$OUTPUT_FILE" \
        task="$TASK" \
        camera_groups="$GGROUP" \
        num_demos=20 \
        save_video=true \
        "$@"

echo "Done at: $(date)"
