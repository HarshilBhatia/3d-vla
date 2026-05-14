#!/bin/bash
# Array job: one GPU per (model, cfg_scale) pair.
# 2 models × 3 cfg scales = 6 tasks.
# model_idx = task_id // 3,  cfg_idx = task_id % 3
#
# Usage:
#   sbatch scripts/eval/eval_final_cfg.slurm
#
#SBATCH --job-name=final_cfg
#SBATCH --array=0-5
#SBATCH --output=logs/final_cfg_%A_%a.out
#SBATCH --error=logs/final_cfg_%A_%a.err
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

MODELS=(single_group multi_group)
CFG_SCALES=(0 1 2)

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
CFG_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

MODEL="${MODELS[$MODEL_IDX]}"
CFG_SCALE="${CFG_SCALES[$CFG_IDX]}"

CHECKPOINT=train_logs/final/siglip_${MODEL}/best.pth
RESULT_DIR=results/final_cfg/siglip_${MODEL}
OUTPUT_CSV=${RESULT_DIR}/cfg_${CFG_SCALE}.csv

mkdir -p logs $RESULT_DIR

if [ "$MODEL" = "single_group" ]; then
    DATA_PATHS=/grogu/user/harshilb/open_drawer/single/train.zarr,/grogu/user/harshilb/open_drawer/val.zarr
else
    DATA_PATHS=/grogu/user/harshilb/open_drawer/multi/train.zarr,/grogu/user/harshilb/open_drawer/val.zarr
fi

echo "Model:      siglip_${MODEL}"
echo "Checkpoint: $CHECKPOINT"
echo "CFG scale:  $CFG_SCALE"
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
