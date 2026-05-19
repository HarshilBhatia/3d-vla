#!/bin/bash
# Per-camera-group eval for G3G4_finetune_default_miscal.
#
# Array layout (2 noise conditions × 2 datasets = 4 jobs):
#   0  train.zarr  clean
#   1  train.zarr  medium G1+G2
#   2  val.zarr    clean
#   3  val.zarr    medium G1+G2
#
# Usage:
#   sbatch scripts/eval/eval_default_miscal.slurm
#SBATCH --job-name=eval_default_miscal
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/eval_default_miscal_%A_%a.out
#SBATCH --error=logs/eval_default_miscal_%A_%a.err
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,14],grogu-3-5,grogu-3-25

set -euo pipefail
echo "Job started at: $(date)  Node: $(hostname)  Array: $SLURM_ARRAY_TASK_ID"

cd /home/harshilb/3d_flowmatch_actor
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

RUN_NAME=G3G4_finetune_default_miscal
DATA_PATHS=(
    /grogu/user/harshilb/multi_cam/train.zarr
    /grogu/user/harshilb/multi_cam/train.zarr
    /grogu/user/harshilb/multi_cam/val.zarr
    /grogu/user/harshilb/multi_cam/val.zarr
)
NOISE_LEVELS=( "" medium  "" medium )
DATA_TAGS=( train train val val )

DATA_PATH="${DATA_PATHS[$SLURM_ARRAY_TASK_ID]}"
NOISE="${NOISE_LEVELS[$SLURM_ARRAY_TASK_ID]}"
DATA_TAG="${DATA_TAGS[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT="train_logs/final/${RUN_NAME}/last.pth"

if [[ -z "$NOISE" ]]; then
    TAG="${DATA_TAG}_clean"
    NOISE_ARGS=()
else
    TAG="${DATA_TAG}_${NOISE}_G1G2"
    NOISE_ARGS=( "cotrain_miscal_group_ids=[1,2]" "cotrain_miscal_level=${NOISE}" )
fi

OUTPUT_CSV="results/per_camera_group_v2/${RUN_NAME}/${TAG}_last.csv"
mkdir -p "$(dirname "$OUTPUT_CSV")" logs

echo "Run: $RUN_NAME  Data: $DATA_TAG  Noise: ${NOISE:-clean}  Checkpoint: $CHECKPOINT"

python scripts/eval/eval_checkpoints_mse_per_camera_group.py \
    checkpoints="$CHECKPOINT" \
    data_paths="$DATA_PATH" \
    val_instructions=instructions/peract/instructions.json \
    bimanual=false \
    output_csv="$OUTPUT_CSV" \
    num_batches=200 \
    "${NOISE_ARGS[@]}" \
    "$@"

echo "Done at: $(date)"
