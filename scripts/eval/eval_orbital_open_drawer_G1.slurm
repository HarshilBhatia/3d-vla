#!/bin/bash
# Array job: 3 checkpoints × 2 runs = indices 0-5
#   ckpt_idx = ARRAY_ID // 2  (0,1,2)
#   run_idx  = ARRAY_ID %  2  (0,1)
#
# Usage:
#   sbatch scripts/eval/eval_orbital_open_drawer_G1.slurm
#   sbatch --array=0-1 ...   # just first checkpoint
#SBATCH --job-name=eval_orb_G1
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/eval_orb_G1_%A_%a.out
#SBATCH --error=logs/eval_orb_G1_%A_%a.err
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,14],grogu-3-5

set -euo pipefail
ulimit -c 0

echo "Job started at: $(date)"
echo "Node: $(hostname)  Array index: $SLURM_ARRAY_TASK_ID"

HOST_REPO=/home/harshilb/3d_flowmatch_actor
CONTAINER_REPO=/root/3d_flowmatch_actor
CONTAINER=/home/harshilb/my_eval_env
COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04

CHECKPOINTS=(
    /root/3d_flowmatch_actor/train_logs/Orbital/open_drawer_default_G1/best.pth
    /root/3d_flowmatch_actor/train_logs/Orbital/open_drawer_multi_groups/best.pth
    /root/3d_flowmatch_actor/train_logs/Orbital/3dfa_run/step_360000.pth
)

RUN_NAMES=(
    new_test
    multi_groups
    3dfa_run_overfit
)

CKPT_IDX=$(( SLURM_ARRAY_TASK_ID ))
RUN_IDX=0

# $(( SLURM_ARRAY_TASK_ID % 2 ))

CHECKPOINT="${CHECKPOINTS[$CKPT_IDX]}"
RUN_NAME="${RUN_NAMES[$CKPT_IDX]}"

OUTPUT_DIR="${HOST_REPO}/eval_logs/Orbital/${RUN_NAME}/multi_cam_dataset_G3/"
OUTPUT_FILE="${OUTPUT_DIR}/new_results_open_drawer_G1_run${RUN_IDX}.json"

mkdir -p "$OUTPUT_DIR" logs

echo "Checkpoint: $CHECKPOINT"
echo "Run:        $RUN_IDX"
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
    bash ${CONTAINER_REPO}/scripts/eval/eval_orbital.sh \
        checkpoint="$CHECKPOINT" \
        output_file="$OUTPUT_FILE" \
        task="open_drawer" \
        data_dir="/grogu/user/harshilb/open_drawer_multi_cam/" \
        spawn_camera_group=G3 \
        save_video=true
        

echo "Done at: $(date)"
