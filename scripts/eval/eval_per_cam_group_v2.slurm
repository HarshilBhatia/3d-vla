#!/bin/bash
# Per-camera-group eval — v2 (bug fix: camera_group reshaped to (B,) before indexing).
#
# Array layout (4 models × 3 noise conditions = 12 jobs):
#   0  multi_cam_G3G4             clean
#   1  multi_cam_G3G4             small  G1+G2
#   2  multi_cam_G3G4             medium G1+G2
#   3  cotrain_mixed_miscal       clean
#   4  cotrain_mixed_miscal       small  G1+G2
#   5  cotrain_mixed_miscal       medium G1+G2
#   6  G3G4_finetune_3dfa         clean
#   7  G3G4_finetune_3dfa         small  G1+G2
#   8  G3G4_finetune_3dfa         medium G1+G2
#   9  G3G4_finetune_deltaM_eef_final   clean
#  10  G3G4_finetune_deltaM_eef_final   small  G1+G2
#  11  G3G4_finetune_deltaM_eef_final   medium G1+G2
#
# Usage:
#   sbatch scripts/eval/eval_per_cam_group_v2.slurm
#SBATCH --job-name=eval_pcg_v2
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/eval_pcg_v2_%A_%a.out
#SBATCH --error=logs/eval_pcg_v2_%A_%a.err
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,14],grogu-3-5,grogu-3-25

set -euo pipefail
echo "Job started at: $(date)  Node: $(hostname)  Array: $SLURM_ARRAY_TASK_ID"

cd /home/harshilb/3d_flowmatch_actor
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

RUN_NAMES=(
    multi_cam_G3G4           multi_cam_G3G4           multi_cam_G3G4
    cotrain_mixed_miscal     cotrain_mixed_miscal     cotrain_mixed_miscal
    G3G4_finetune_3dfa       G3G4_finetune_3dfa       G3G4_finetune_3dfa
    G3G4_finetune_deltaM_eef_final  G3G4_finetune_deltaM_eef_final  G3G4_finetune_deltaM_eef_final
)
NOISE_LEVELS=( "" small medium  "" small medium  "" small medium  "" small medium )

RUN_NAME="${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}"
NOISE="${NOISE_LEVELS[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT="train_logs/final/${RUN_NAME}/last.pth"
DATA_PATH="/grogu/user/harshilb/multi_cam/train.zarr"

if [[ -z "$NOISE" ]]; then
    TAG="clean"
    NOISE_ARGS=()
else
    TAG="${NOISE}_G1G2"
    NOISE_ARGS=( "cotrain_miscal_group_ids=[1,2]" "cotrain_miscal_level=${NOISE}" )
fi

OUTPUT_CSV="results/per_camera_group_v2/${RUN_NAME}/${TAG}_last.csv"
mkdir -p "$(dirname "$OUTPUT_CSV")" logs

echo "Run: $RUN_NAME  Noise: ${TAG}  Checkpoint: $CHECKPOINT"

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
