#!/bin/bash
# Array job: one GPU per CFG scale.
# Evaluates the open_drawer_multi_groups_clip_langdrop20 best checkpoint
# across the 3 zarr datasets, for each cfg_scale in {0,1,2,3,4,5,6,7,8,10,100}.
# cfg_scale=0 is unconditional generation (CFG formula collapses to out_uncond).
#
# Usage:
#   sbatch scripts/eval/eval_checkpoints_mse_cfg.slurm
#
#SBATCH --job-name=ckpt_mse_cfg
#SBATCH --array=0-10
#SBATCH --output=logs/ckpt_mse_cfg_%A_%a.out
#SBATCH --error=logs/ckpt_mse_cfg_%A_%a.err
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

RESULT_DIR=results/checkpoint_mse_cfg/open_drawer_multi_groups_clip_langdrop20
mkdir -p logs $RESULT_DIR

CHECKPOINT=train_logs/Orbital/open_drawer_multi_groups_clip_langdrop20/best.pth

# ── One CFG scale per array index ─────────────────────────────────────────────
# Index 0 → cfg=0 (unconditional). Indices 1..10 → cfg ∈ {1,2,...,8,10,100}.
CFG_SCALES=(0 1 2 3 4 5 6 7 8 10 100)
CFG_SCALE="${CFG_SCALES[$SLURM_ARRAY_TASK_ID]}"

OUTPUT_CSV=${RESULT_DIR}/cfg_${CFG_SCALE}.csv

# ── Zarr data paths to eval on (comma-separated, no spaces) ───────────────────
DATA_PATHS=\
/grogu/user/harshilb/1task_new.zarr,\
/grogu/user/harshilb/open_drawer_G1_new_seed/train.zarr,\
/grogu/user/harshilb/G3_val_set/val.zarr

echo "Checkpoint: $CHECKPOINT"
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
