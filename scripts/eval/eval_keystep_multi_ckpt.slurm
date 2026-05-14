#!/bin/bash
# Array job: one task per checkpoint — plain conditional keystep eval.
#   index 0 → open_drawer_multi_groups
#   index 1 → siglip_single_cam
#
# Usage:
#   sbatch scripts/eval/eval_keystep_multi_ckpt.slurm
#
#SBATCH --job-name=keystep_multi
#SBATCH --array=0-2
#SBATCH --output=logs/keystep_multi_%A_%a.out
#SBATCH --error=logs/keystep_multi_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=shubhamlong
#SBATCH --exclude=grogu-0-[14,19,29],grogu-1-[6,9,14],grogu-3-25

set -euo pipefail

echo "Job started at: $(date)"
echo "Node: $(hostname)  Array index: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"

cd /home/harshilb/3d_flowmatch_actor
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

# ── One checkpoint per array index ────────────────────────────────────────────
CKPT_NAMES=(
    open_drawer_multi_groups
    siglip_single_cam
    open_drawer_default_G1
)

CKPT_NAME="${CKPT_NAMES[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT=train_logs/Orbital/${CKPT_NAME}/best.pth

RESULT_DIR=results/keystep/${CKPT_NAME}
mkdir -p logs "$RESULT_DIR"

OUTPUT_CSV=${RESULT_DIR}/conditional.csv

DATA_PATHS=\
/grogu/user/harshilb/1task_new.zarr,\
/grogu/user/harshilb/open_drawer_G1_new_seed/train.zarr,\
/grogu/user/harshilb/G3_val_set/val.zarr

echo "Checkpoint:  $CHECKPOINT"
echo "Output CSV:  $OUTPUT_CSV"

python scripts/eval/eval_checkpoints_mse_cfg_keystep.py \
    checkpoints=$CHECKPOINT \
    data_paths=$DATA_PATHS \
    val_instructions=instructions/peract/instructions.json \
    bimanual=false \
    num_batches=200 \
    batch_size_val=32 \
    output_csv=$OUTPUT_CSV \
    "$@"

echo "Done at: $(date)"
