#!/bin/bash
# Array job: one GPU per checkpoint from train_logs/Orbital/siglip_single_cam/
#
# Usage:
#   sbatch scripts/eval/eval_checkpoints_mse.slurm
#   sbatch scripts/eval/eval_checkpoints_mse.slurm wandb_project=my_project
#
#SBATCH --job-name=ckpt_mse_eval
#SBATCH --array=0-6
#SBATCH --output=logs/ckpt_mse_eval_%A_%a.out
#SBATCH --error=logs/ckpt_mse_eval_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --partition=shubhamlong
#SBATCH --exclude=grogu-0-[14,19,29],grogu-1-[6,9,14],grogu-3-25

set -euo pipefail

echo "Job started at: $(date)"
echo "Node: $(hostname)  Array index: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"

cd /home/harshilb/3d_flowmatch_actor
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

RESULT_DIR=results/checkpoint_mse/siglip_fps
mkdir -p logs $RESULT_DIR

# ── One checkpoint per array index ────────────────────────────────────────────
CKPT_DIR=train_logs/Orbital/siglip_nofps
CHECKPOINTS=(
    ${CKPT_DIR}/interm_step_10000.pth
    ${CKPT_DIR}/interm_step_20000.pth
    ${CKPT_DIR}/interm_step_30000.pth
    ${CKPT_DIR}/interm_step_40000.pth
    ${CKPT_DIR}/interm_step_50000.pth
    ${CKPT_DIR}/interm_step_60000.pth
    ${CKPT_DIR}/interm_step_70000.pth
)
CHECKPOINT="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"

# Per-checkpoint CSV — each array task writes its own file so parallel tasks
# never race on the same file. plot_checkpoint_mse.py aggregates them.
STEP=$(basename $CHECKPOINT .pth | grep -oP '\d+$')
OUTPUT_CSV=${RESULT_DIR}/step_${STEP}.csv

# ── Zarr data paths to eval on (comma-separated, no spaces) ───────────────────
DATA_PATHS=\
/grogu/user/harshilb/1task_new.zarr,\
/grogu/user/harshilb/open_drawer_G1_new_seed/train.zarr,\
/grogu/user/harshilb/G3_val_set/val.zarr

echo "Checkpoint: $CHECKPOINT  (step $STEP)"
echo "Data paths: $DATA_PATHS"
echo "Output CSV: $OUTPUT_CSV"

python scripts/eval/eval_checkpoints_mse.py \
    checkpoints=$CHECKPOINT \
    data_paths=$DATA_PATHS \
    val_instructions=instructions/peract/instructions.json \
    bimanual=false \
    backbone=siglip2 \
    num_batches=100 \
    batch_size_val=32 \
    output_csv=$OUTPUT_CSV \
    "$@"

echo "Done at: $(date)"
