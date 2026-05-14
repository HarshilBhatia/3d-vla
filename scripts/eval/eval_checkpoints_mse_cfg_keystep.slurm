#!/bin/bash
# Array job: one GPU per cfg_scale in {0, 1, 2, 5, 10}.
# Reports MSE metrics bucketed by keypose step (0=start, 1=mid, 2=end).
#
# Usage:
#   sbatch scripts/eval/eval_checkpoints_mse_cfg_keystep.slurm
#
#SBATCH --job-name=ckpt_keystep
#SBATCH --array=0-4
#SBATCH --output=logs/ckpt_keystep_%A_%a.out
#SBATCH --error=logs/ckpt_keystep_%A_%a.err
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

RESULT_DIR=results/keystep/open_drawer_multi_groups_clip_langdrop20
mkdir -p logs $RESULT_DIR

CHECKPOINT=train_logs/Orbital/open_drawer_multi_groups_clip_langdrop20/best.pth

CFG_SCALES=(0 1 2 5 10)
CFG_SCALE="${CFG_SCALES[$SLURM_ARRAY_TASK_ID]}"

OUTPUT_CSV=${RESULT_DIR}/cfg_${CFG_SCALE}.csv

DATA_PATHS=\
/grogu/user/harshilb/1task_new.zarr,\
/grogu/user/harshilb/open_drawer_G1_new_seed/train.zarr,\
/grogu/user/harshilb/G3_val_set/val.zarr

echo "Checkpoint:  $CHECKPOINT"
echo "CFG scale:   $CFG_SCALE"
echo "Output CSV:  $OUTPUT_CSV"

python scripts/eval/eval_checkpoints_mse_cfg_keystep.py \
    checkpoints=$CHECKPOINT \
    data_paths=$DATA_PATHS \
    val_instructions=instructions/peract/instructions.json \
    bimanual=false \
    cfg_scale=$CFG_SCALE \
    num_batches=200 \
    batch_size_val=32 \
    output_csv=$OUTPUT_CSV \
    "$@"

echo "Done at: $(date)"
