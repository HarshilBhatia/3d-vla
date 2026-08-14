#!/bin/bash
# Online eval on turn_tap for delta checkpoints × 6 random miscal levels, G5 only.
#
# Array layout (run_idx = idx / 6, noise_idx = idx % 6):
#   runs:   default, deltaM, deltaM_6x6
#   noise:  none | 2deg+2cm | 5deg+5cm | 10deg+10cm | 15deg+15cm | 20deg+20cm
#
#   idx 0  -> default    / none
#   idx 1  -> default    / 2deg+2cm
#   idx 2  -> default    / 5deg+5cm
#   idx 3  -> default    / 10deg+10cm
#   idx 4  -> default    / 15deg+15cm
#   idx 5  -> default    / 20deg+20cm
#   idx 6  -> deltaM     / none
#   idx 7  -> deltaM     / 2deg+2cm
#   idx 8  -> deltaM     / 5deg+5cm
#   idx 9  -> deltaM     / 10deg+10cm
#   idx 10 -> deltaM     / 15deg+15cm
#   idx 11 -> deltaM     / 20deg+20cm
#   idx 12 -> deltaM_6x6 / none
#   idx 13 -> deltaM_6x6 / 2deg+2cm
#   idx 14 -> deltaM_6x6 / 5deg+5cm
#   idx 15 -> deltaM_6x6 / 10deg+10cm
#   idx 16 -> deltaM_6x6 / 15deg+15cm
#   idx 17 -> deltaM_6x6 / 20deg+20cm
#
# Usage:
#   sbatch scripts/eval/online_eval_delta_randmiscal_turn_tap_G5.slurm
#SBATCH --job-name=eval_delta_randmiscal_turn_tap_G5
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/eval_delta_randmiscal_turn_tap_G5_%A_%a.out
#SBATCH --error=logs/eval_delta_randmiscal_turn_tap_G5_%A_%a.err
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,14],grogu-2-22,grogu-3-5,grogu-3-25

set -euo pipefail
ulimit -c 0

echo "Job started at: $(date)"
echo "Node: $(hostname)  Array index: $SLURM_ARRAY_TASK_ID"

HOST_REPO=/home/harshilb/3d_flowmatch_actor
CONTAINER_REPO=/root/3d_flowmatch_actor
CONTAINER=/home/harshilb/my_eval_env
COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04

RUN_NAMES=(default deltaM deltaM_6x6)
NOISE_LABELS=(none 2deg2cm 5deg5cm 10deg10cm 15deg15cm 20deg20cm)
ROT_LEVELS=("" "2deg" "5deg" "10deg" "15deg" "20deg")
TRANS_LEVELS=("" "2cm" "5cm" "10cm" "15cm" "20cm")

RUN_IDX=$(( SLURM_ARRAY_TASK_ID / 6 ))
NOISE_IDX=$(( SLURM_ARRAY_TASK_ID % 6 ))

RUN_NAME="${RUN_NAMES[$RUN_IDX]}"
NOISE_LABEL="${NOISE_LABELS[$NOISE_IDX]}"
ROT="${ROT_LEVELS[$NOISE_IDX]}"
TRANS="${TRANS_LEVELS[$NOISE_IDX]}"
GGROUP="G5"
TASK="turn_tap"

CHECKPOINT="${CONTAINER_REPO}/train_logs/delta/${RUN_NAME}/interm_step_160000.pth"
OUTPUT_FILE="${HOST_REPO}/eval_logs/20may_delta_randmiscal/${RUN_NAME}/${GGROUP}/${NOISE_LABEL}/results_${TASK}.json"

mkdir -p "$(dirname "$OUTPUT_FILE")" logs

echo "Run:        $RUN_NAME"
echo "Noise:      $NOISE_LABEL  (rot=$ROT  trans=$TRANS)"
echo "GGroup:     $GGROUP"
echo "Checkpoint: $CHECKPOINT"
echo "Output:     $OUTPUT_FILE"

MISCAL_ARGS=()
[ -n "$ROT" ]   && MISCAL_ARGS+=("++miscal_rot_level=$ROT")
[ -n "$TRANS" ] && MISCAL_ARGS+=("++miscal_trans_level=$TRANS")

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
        save_video=true \
        "${MISCAL_ARGS[@]}" \
        "$@"

echo "Done at: $(date)"
