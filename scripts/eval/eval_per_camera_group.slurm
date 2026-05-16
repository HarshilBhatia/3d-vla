#!/bin/bash
# Offline MSE eval grouped by camera_group, on each run's training zarr.
#
# Array layout (noise eval perturbs ALL groups present in the dataset):
#   0 -> multi_cam_G3G4         on multi_cam_G3G4.zarr           (clean)
#   1 -> cotrain_mixed_miscal   on multi_cam/train.zarr          (clean)
#   2 -> cotrain_mixed_miscal   on multi_cam/train.zarr          (small noise on G1-G4)
#   3 -> cotrain_mixed_miscal   on multi_cam/train.zarr          (medium noise on G1-G4)
#   4 -> multi_cam_G3G4         on multi_cam/train.zarr          (clean)
#   5 -> multi_cam_G3G4         on multi_cam/train.zarr          (small noise on G1-G4)
#   6 -> multi_cam_G3G4         on multi_cam/train.zarr          (medium noise on G1-G4)
#   7 -> cotrain_mixed_miscal   on multi_cam/train.zarr          (clean, no deltaM rope)
#   8 -> cotrain_mixed_miscal   on multi_cam/train.zarr          (small noise G1-G4, no deltaM rope)
#   9 -> cotrain_mixed_miscal   on multi_cam/train.zarr          (medium noise G1-G4, no deltaM rope)
#
# Usage:
#   sbatch scripts/eval/eval_per_camera_group.slurm
#   sbatch --array=7-9 scripts/eval/eval_per_camera_group.slurm
#SBATCH --job-name=eval_per_cam_group
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/eval_per_cam_group_%A_%a.out
#SBATCH --error=logs/eval_per_cam_group_%A_%a.err
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,14],grogu-3-5,grogu-3-25

set -euo pipefail

echo "Job started at: $(date)"
echo "Node: $(hostname)  Array index: $SLURM_ARRAY_TASK_ID"

cd /home/harshilb/3d_flowmatch_actor
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

RUN_NAMES=(multi_cam_G3G4 cotrain_mixed_miscal cotrain_mixed_miscal cotrain_mixed_miscal multi_cam_G3G4 multi_cam_G3G4 multi_cam_G3G4 cotrain_mixed_miscal cotrain_mixed_miscal cotrain_mixed_miscal)
DATA_PATHS=(
    /grogu/user/harshilb/multi_cam_G3G4.zarr
    /grogu/user/harshilb/multi_cam/train.zarr
    /grogu/user/harshilb/multi_cam/train.zarr
    /grogu/user/harshilb/multi_cam/train.zarr
    /grogu/user/harshilb/multi_cam/train.zarr
    /grogu/user/harshilb/multi_cam/train.zarr
    /grogu/user/harshilb/multi_cam/train.zarr
    /grogu/user/harshilb/multi_cam/train.zarr
    /grogu/user/harshilb/multi_cam/train.zarr
    /grogu/user/harshilb/multi_cam/train.zarr
)
MISCAL_LEVELS=("" "" small medium "" small medium "" small medium)   # "" means clean
TAGS=(clean clean small medium clean_on_multi_cam small_on_multi_cam medium_on_multi_cam clean_no_deltaM small_no_deltaM medium_no_deltaM)
NO_DELTAM=(0 0 0 0 0 0 0 1 1 1)   # 1 = pass predict_extrinsics=false

RUN_NAME="${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}"
DATA_PATH="${DATA_PATHS[$SLURM_ARRAY_TASK_ID]}"
MISCAL_LEVEL="${MISCAL_LEVELS[$SLURM_ARRAY_TASK_ID]}"
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"
USE_NO_DELTAM="${NO_DELTAM[$SLURM_ARRAY_TASK_ID]}"

CHECKPOINT="train_logs/Orbital/${RUN_NAME}/last.pth"
OUTPUT_CSV="results/per_camera_group/${RUN_NAME}/${TAG}_last.csv"

mkdir -p "$(dirname "$OUTPUT_CSV")" logs

echo "Run:        $RUN_NAME"
echo "Data:       $DATA_PATH"
echo "Noise:      ${MISCAL_LEVEL:-none}  (tag=$TAG)"
echo "Checkpoint: $CHECKPOINT"
echo "Output CSV: $OUTPUT_CSV"

EXTRA_ARGS=()
if [[ -n "$MISCAL_LEVEL" ]]; then
    EXTRA_ARGS+=( "cotrain_miscal_group_ids=[1,2,3,4]" "cotrain_miscal_level=$MISCAL_LEVEL" )
fi
if [[ "$USE_NO_DELTAM" == "1" ]]; then
    EXTRA_ARGS+=( "predict_extrinsics=false" )
fi

python scripts/eval/eval_checkpoints_mse_per_camera_group.py \
    checkpoints="$CHECKPOINT" \
    data_paths="$DATA_PATH" \
    val_instructions=instructions/peract/instructions.json \
    bimanual=false \
    output_csv="$OUTPUT_CSV" \
    num_batches=200 \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "Done at: $(date)"
