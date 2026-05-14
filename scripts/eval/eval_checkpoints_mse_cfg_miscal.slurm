#!/bin/bash
# Array job: one GPU per (cfg_scale, miscal_level) pair.
# Sweeps cfg ∈ {0,1,2,3,4,5,6,7,8,10,100} × noise ∈ {small,medium,large} = 33 tasks.
# cfg=0 is unconditional generation (CFG formula collapses to out_uncond).
#
# Layout: index = noise_idx * 11 + cfg_idx
#   noise_idx 0 → small,  cfg_idx 0..10 → indices 0..10
#   noise_idx 1 → medium, cfg_idx 0..10 → indices 11..21
#   noise_idx 2 → large,  cfg_idx 0..10 → indices 22..32
#
# Usage:
#   sbatch scripts/eval/eval_checkpoints_mse_cfg_miscal.slurm
#   # one cell only:
#   sbatch --array=15 scripts/eval/eval_checkpoints_mse_cfg_miscal.slurm
#
#SBATCH --job-name=ckpt_mse_cfg_mis
#SBATCH --array=0-32
#SBATCH --output=logs/ckpt_mse_cfg_mis_%A_%a.out
#SBATCH --error=logs/ckpt_mse_cfg_mis_%A_%a.err
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

RESULT_DIR=results/checkpoint_mse_cfg_miscal/open_drawer_multi_groups_clip_langdrop20
mkdir -p logs $RESULT_DIR

CHECKPOINT=train_logs/Orbital/open_drawer_multi_groups_clip_langdrop20/best.pth

# ── Decode (noise_level, cfg_scale) from $SLURM_ARRAY_TASK_ID ─────────────────
CFG_SCALES=(0 1 2 3 4 5 6 7 8 10 100)
NOISE_LEVELS=(small medium large)
NUM_CFG=${#CFG_SCALES[@]}     # 11

NOISE_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_CFG ))
CFG_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_CFG ))

NOISE_LEVEL="${NOISE_LEVELS[$NOISE_IDX]}"
CFG_SCALE="${CFG_SCALES[$CFG_IDX]}"

OUTPUT_CSV=${RESULT_DIR}/cfg_${CFG_SCALE}_${NOISE_LEVEL}.csv

# ── Zarr data paths to eval on (comma-separated, no spaces) ───────────────────
DATA_PATHS=\
/grogu/user/harshilb/1task_new.zarr,\
/grogu/user/harshilb/open_drawer_G1_new_seed/train.zarr,\
/grogu/user/harshilb/G3_val_set/val.zarr

echo "Checkpoint:  $CHECKPOINT"
echo "CFG scale:   $CFG_SCALE"
echo "Noise level: $NOISE_LEVEL"
echo "Data paths:  $DATA_PATHS"
echo "Output CSV:  $OUTPUT_CSV"

python scripts/eval/eval_checkpoints_mse_cfg_miscal.py \
    checkpoints=$CHECKPOINT \
    data_paths=$DATA_PATHS \
    val_instructions=instructions/peract/instructions.json \
    bimanual=false \
    cfg_scale=$CFG_SCALE \
    orbital_miscal_noise_level=$NOISE_LEVEL \
    num_batches=100 \
    batch_size_val=32 \
    output_csv=$OUTPUT_CSV \
    "$@"

echo "Done at: $(date)"
