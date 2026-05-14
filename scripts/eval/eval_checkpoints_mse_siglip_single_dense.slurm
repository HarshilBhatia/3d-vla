#!/bin/bash
# Array job: one GPU per (checkpoint, cfg_scale) pair.
# 7 checkpoints × 6 cfg scales = 42 tasks.
# cfg_idx = task_id // 7,  ckpt_idx = task_id % 7
#
# Usage:
#   sbatch scripts/eval/eval_checkpoints_mse_siglip_single_dense.slurm
#
#SBATCH --job-name=ckpt_mse_s_dense
#SBATCH --array=0-41
#SBATCH --output=logs/ckpt_mse_s_dense_%A_%a.out
#SBATCH --error=logs/ckpt_mse_s_dense_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --partition=all
#SBATCH --exclude=grogu-0-[14,19,29],grogu-1-[6,9,14],grogu-2-22,grogu-3-25

set -euo pipefail

echo "Job started at: $(date)"
echo "Node: $(hostname)  Array index: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"

cd /home/harshilb/3d_flowmatch_actor
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

CKPT_DIR=train_logs/Orbital/siglip_single_dense
CHECKPOINTS=(
    ${CKPT_DIR}/interm_step_10000.pth
    ${CKPT_DIR}/interm_step_20000.pth
    ${CKPT_DIR}/interm_step_30000.pth
    ${CKPT_DIR}/interm_step_40000.pth
    ${CKPT_DIR}/interm_step_50000.pth
    ${CKPT_DIR}/interm_step_60000.pth
    ${CKPT_DIR}/interm_step_70000.pth
)
CFG_SCALES=(0 1 2 4 6 8)

CKPT_IDX=$(( SLURM_ARRAY_TASK_ID % 7 ))
CFG_IDX=$(( SLURM_ARRAY_TASK_ID / 7 ))

CHECKPOINT="${CHECKPOINTS[$CKPT_IDX]}"
CFG_SCALE="${CFG_SCALES[$CFG_IDX]}"

STEP=$(basename $CHECKPOINT .pth | grep -oP '\d+$')
RESULT_DIR=results/checkpoint_mse_cfg/siglip_single_dense/cfg_${CFG_SCALE}
OUTPUT_CSV=${RESULT_DIR}/step_${STEP}.csv

mkdir -p logs $RESULT_DIR

DATA_PATHS=\
/grogu/user/harshilb/dense/open_drawer_multi_cam/train.zarr,\
/grogu/user/harshilb/dense/G3_val.zarr

echo "Checkpoint: $CHECKPOINT  (step $STEP)"
echo "CFG scale:  $CFG_SCALE"
echo "Data paths: $DATA_PATHS"
echo "Output CSV: $OUTPUT_CSV"

python scripts/eval/eval_checkpoints_mse_cfg.py \
    checkpoints=$CHECKPOINT \
    data_paths=$DATA_PATHS \
    val_instructions=instructions/peract/instructions.json \
    bimanual=false \
    cfg_scale=$CFG_SCALE \
    num_batches=100 \
    batch_size_val=32 \
    output_csv=$OUTPUT_CSV \
    "$@"

echo "Done at: $(date)"
