#!/bin/bash
# Per-camera-group eval for fine-tuned runs.
#
# Array layout:
#   0 -> G3G4_finetune_3dfa          medium noise on G1+G2
#   1 -> G3G4_finetune_deltaM_eef_final  medium noise on G1+G2
#
# Usage:
#   sbatch scripts/eval/eval_per_cam_group_finetune.slurm
#SBATCH --job-name=eval_pcg_ft
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/eval_pcg_ft_%A_%a.out
#SBATCH --error=logs/eval_pcg_ft_%A_%a.err
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,14],grogu-3-5,grogu-3-25

set -euo pipefail

echo "Job started at: $(date)"
echo "Node: $(hostname)  Array index: $SLURM_ARRAY_TASK_ID"

cd /home/harshilb/3d_flowmatch_actor
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

RUN_NAMES=(G3G4_finetune_3dfa G3G4_finetune_deltaM_eef_final)

RUN_NAME="${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT="train_logs/final/${RUN_NAME}/last.pth"
DATA_PATH="/grogu/user/harshilb/multi_cam/train.zarr"
OUTPUT_CSV="results/per_camera_group/${RUN_NAME}/medium_G1G2_last.csv"

mkdir -p "$(dirname "$OUTPUT_CSV")" logs

echo "Run:        $RUN_NAME"
echo "Checkpoint: $CHECKPOINT"
echo "Data:       $DATA_PATH"
echo "Output CSV: $OUTPUT_CSV"

python scripts/eval/eval_checkpoints_mse_per_camera_group.py \
    checkpoints="$CHECKPOINT" \
    data_paths="$DATA_PATH" \
    val_instructions=instructions/peract/instructions.json \
    bimanual=false \
    output_csv="$OUTPUT_CSV" \
    num_batches=200 \
    "cotrain_miscal_group_ids=[1,2]" \
    cotrain_miscal_level=medium \
    "$@"

echo "Done at: $(date)"
