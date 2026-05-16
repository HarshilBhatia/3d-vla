#!/bin/bash
# Sweep fixed-magnitude miscal noise (rot=0..10deg, trans=0..10cm) on G1/G3
# samples of the training zarr, for two checkpoints. Array layout:
#
#   index  ->  ckpt                       sweep_axis
#     0    ->  cotrain_mixed_miscal       diagonal
#     1    ->  cotrain_mixed_miscal       rot
#     2    ->  cotrain_mixed_miscal       trans
#     3    ->  multi_cam_G3G4             diagonal
#     4    ->  multi_cam_G3G4             rot
#     5    ->  multi_cam_G3G4             trans
#
# Each task writes its own CSV under results/miscal_sweep/G1G3/.
# Merge with: `cat results/miscal_sweep/G1G3/*.csv > merged.csv` (then dedupe headers).
#
# Usage:
#   sbatch scripts/eval/eval_miscal_sweep_G1G3.slurm
#   sbatch --array=0,3 scripts/eval/eval_miscal_sweep_G1G3.slurm   # just the diagonals
#
#SBATCH --job-name=eval_miscal_sweep_G1G3
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --array=0-5
#SBATCH --output=logs/eval_miscal_sweep_G1G3_%A_%a.out
#SBATCH --error=logs/eval_miscal_sweep_G1G3_%A_%a.err
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,14],grogu-3-5,grogu-3-25

set -euo pipefail

echo "Job started at: $(date)"
echo "Node: $(hostname)  Array index: $SLURM_ARRAY_TASK_ID"

cd /home/harshilb/3d_flowmatch_actor
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

CKPT_PATHS=(
    train_logs/Orbital/cotrain_mixed_miscal/last.pth
    train_logs/Orbital/cotrain_mixed_miscal/last.pth
    train_logs/Orbital/cotrain_mixed_miscal/last.pth
    train_logs/Orbital/multi_cam_G3G4/last.pth
    train_logs/Orbital/multi_cam_G3G4/last.pth
    train_logs/Orbital/multi_cam_G3G4/last.pth
)
SWEEP_AXES=(diagonal rot trans diagonal rot trans)
CKPT_NAMES=(cotrain_mixed_miscal cotrain_mixed_miscal cotrain_mixed_miscal multi_cam_G3G4 multi_cam_G3G4 multi_cam_G3G4)

CKPT="${CKPT_PATHS[$SLURM_ARRAY_TASK_ID]}"
AXIS="${SWEEP_AXES[$SLURM_ARRAY_TASK_ID]}"
CKPT_NAME="${CKPT_NAMES[$SLURM_ARRAY_TASK_ID]}"

DATA_PATH="/grogu/user/harshilb/multi_cam/train.zarr"
OUTPUT_CSV="results/miscal_sweep/G1G3/${CKPT_NAME}_${AXIS}.csv"

mkdir -p "$(dirname "$OUTPUT_CSV")" logs

echo "Checkpoint: $CKPT"
echo "Sweep axis: $AXIS"
echo "Data:       $DATA_PATH"
echo "Output CSV: $OUTPUT_CSV"

python scripts/eval/eval_checkpoints_mse_miscal_sweep.py \
    checkpoints="$CKPT" \
    data_paths="$DATA_PATH" \
    val_instructions=instructions/peract/instructions.json \
    bimanual=false \
    output_csv="$OUTPUT_CSV" \
    num_batches=100 \
    max_level=10 \
    sweep_axis="$AXIS" \
    "allowed_groups=[1,3]" \
    seed=0 \
    "$@"

echo "Done at: $(date)"
