#!/bin/bash
# Per-camera-group eval for new runs:
#   cotrain_medium_12_deltaM
#   3dfa_G3G4_finetune_cotrain_miscal
#
# Array layout (2 runs × 3 noise × 2 data = 12 jobs):
#   0  cotrain_medium_12_deltaM          train.zarr  clean
#   1  cotrain_medium_12_deltaM          train.zarr  small  G1+G2
#   2  cotrain_medium_12_deltaM          train.zarr  medium G1+G2
#   3  cotrain_medium_12_deltaM          val.zarr    clean
#   4  cotrain_medium_12_deltaM          val.zarr    small  G1+G2
#   5  cotrain_medium_12_deltaM          val.zarr    medium G1+G2
#   6  3dfa_G3G4_finetune_cotrain_miscal train.zarr  clean
#   7  3dfa_G3G4_finetune_cotrain_miscal train.zarr  small  G1+G2
#   8  3dfa_G3G4_finetune_cotrain_miscal train.zarr  medium G1+G2
#   9  3dfa_G3G4_finetune_cotrain_miscal val.zarr    clean
#  10  3dfa_G3G4_finetune_cotrain_miscal val.zarr    small  G1+G2
#  11  3dfa_G3G4_finetune_cotrain_miscal val.zarr    medium G1+G2
#
# Usage:
#   sbatch scripts/eval/eval_new_runs.slurm
#SBATCH --job-name=eval_new_runs
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/eval_new_runs_%A_%a.out
#SBATCH --error=logs/eval_new_runs_%A_%a.err
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,14],grogu-3-5,grogu-3-25

set -euo pipefail
echo "Job started at: $(date)  Node: $(hostname)  Array: $SLURM_ARRAY_TASK_ID"

cd /home/harshilb/3d_flowmatch_actor
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

RUN_NAMES=(
    cotrain_medium_12_deltaM  cotrain_medium_12_deltaM  cotrain_medium_12_deltaM
    cotrain_medium_12_deltaM  cotrain_medium_12_deltaM  cotrain_medium_12_deltaM
    3dfa_G3G4_finetune_cotrain_miscal  3dfa_G3G4_finetune_cotrain_miscal  3dfa_G3G4_finetune_cotrain_miscal
    3dfa_G3G4_finetune_cotrain_miscal  3dfa_G3G4_finetune_cotrain_miscal  3dfa_G3G4_finetune_cotrain_miscal
)
DATA_PATHS=(
    /grogu/user/harshilb/multi_cam/train.zarr
    /grogu/user/harshilb/multi_cam/train.zarr
    /grogu/user/harshilb/multi_cam/train.zarr
    /grogu/user/harshilb/multi_cam/val.zarr
    /grogu/user/harshilb/multi_cam/val.zarr
    /grogu/user/harshilb/multi_cam/val.zarr
    /grogu/user/harshilb/multi_cam/train.zarr
    /grogu/user/harshilb/multi_cam/train.zarr
    /grogu/user/harshilb/multi_cam/train.zarr
    /grogu/user/harshilb/multi_cam/val.zarr
    /grogu/user/harshilb/multi_cam/val.zarr
    /grogu/user/harshilb/multi_cam/val.zarr
)
NOISE_LEVELS=( "" small medium  "" small medium  "" small medium  "" small medium )
DATA_TAGS=( train train train val val val  train train train val val val )

RUN_NAME="${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}"
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
