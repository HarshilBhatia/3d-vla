#!/bin/bash
# Online eval on open_drawer for delta checkpoints × G1/G5, with medium miscal noise.
#   idx 0 -> default    on G1
#   idx 1 -> default    on G5
#   idx 2 -> deltaM     on G1
#   idx 3 -> deltaM     on G5
#   idx 4 -> deltaM_6x6 on G1
#   idx 5 -> deltaM_6x6 on G5
#
# Usage:
#   sbatch scripts/eval/online_eval_delta_miscal_G1G5.slurm
#SBATCH --job-name=eval_delta_miscal_G1G5
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/eval_delta_miscal_G1G5_%A_%a.out
#SBATCH --error=logs/eval_delta_miscal_G1G5_%A_%a.err
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,14],grogu-3-5,grogu-3-25

set -euo pipefail
ulimit -c 0

echo "Job started at: $(date)"
echo "Node: $(hostname)  Array index: $SLURM_ARRAY_TASK_ID"

HOST_REPO=/home/harshilb/3d_flowmatch_actor
CONTAINER_REPO=/root/3d_flowmatch_actor
CONTAINER=/home/harshilb/my_eval_env
COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04

RUN_NAMES=(default deltaM deltaM_6x6)
MY_GROUPS=(G1 G5)

RUN_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
GROUP_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))

RUN_NAME="${RUN_NAMES[$RUN_IDX]}"
GGROUP="${MY_GROUPS[$GROUP_IDX]}"

TASK="open_drawer"
CHECKPOINT="${CONTAINER_REPO}/train_logs/delta/${RUN_NAME}/interm_step_160000.pth"
OUTPUT_FILE="${HOST_REPO}/eval_logs/20may_delta/${RUN_NAME}/${GGROUP}/results_${TASK}.json"

mkdir -p "$(dirname "$OUTPUT_FILE")" logs

echo "Run:        $RUN_NAME"
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
    bash ${CONTAINER_REPO}/scripts/eval/eval_orbital.sh \
        bimanual=false \
        checkpoint="$CHECKPOINT" \
        output_file="$OUTPUT_FILE" \
        task="$TASK" \
        data_dir="/grogu/user/harshilb/low_dim_demos/" \
        camera_groups="GT" \
        spawn_camera_group="$GGROUP" \
        orbital_miscal_noise_level=medium \
        save_video=true \
        "$@"

echo "Done at: $(date)"
