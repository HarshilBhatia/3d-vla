#!/bin/bash
#SBATCH --job-name=eval_peract
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/eval_peract_%j.out
#SBATCH --error=logs/eval_peract_%j.err
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,14],grogu-3-5

set -euo pipefail
ulimit -c 0

echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

HOST_REPO=/home/harshilb/3d_flowmatch_actor
CONTAINER_REPO=/root/3d_flowmatch_actor
CONTAINER=/home/harshilb/my_eval_env
COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04

TASK="${TASK:-open_drawer}"
CHECKPOINT="${CHECKPOINT:-${CONTAINER_REPO}/train_logs/Peract/peract_collected/step_70000.pth}"
OUTPUT_FILE="${OUTPUT_FILE:-${HOST_REPO}/eval_logs/Peract/peract_collected_new/results_${TASK}.json}"
IMAGE_SIZE="${IMAGE_SIZE:-128,128}"

mkdir -p "$(dirname "$OUTPUT_FILE")" logs

echo "Task:       $TASK"
echo "Checkpoint: $CHECKPOINT"
echo "Output:     $OUTPUT_FILE"
echo "Image size: $IMAGE_SIZE"

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
        dataset=PeractCollected \
        data_dir=/grogu/user/harshilb/peract_rollouts \
        val_instructions=instructions/peract/instructions.json \
        "image_size='${IMAGE_SIZE}'" \
        checkpoint="$CHECKPOINT" \
        output_file="$OUTPUT_FILE" \
        task="$TASK" \
        eval_use_depth2cloud=true \
        save_video=true \
        "$@"

echo "Done at: $(date)"
