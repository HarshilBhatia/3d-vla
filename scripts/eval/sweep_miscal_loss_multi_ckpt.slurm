#!/bin/bash
# Array job: one index per checkpoint
#   ARRAY_ID = checkpoint index (0, 1, 2, ...)
#
# Usage:
#   sbatch scripts/eval/sweep_miscal_loss_multi_ckpt.slurm
#SBATCH --job-name=miscal_sweep
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/miscal_sweep_%A_%a.out
#SBATCH --error=logs/miscal_sweep_%A_%a.err
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,14],grogu-3-5

set -euo pipefail

echo "Job started at: $(date)"
echo "Node: $(hostname)  Array index: $SLURM_ARRAY_TASK_ID"

cd /home/harshilb/3d_flowmatch_actor
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

CHECKPOINTS=(
    
    
)

CHECKPOINT="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"
EXP_NAME=$(basename $(dirname $CHECKPOINT))
OUTPUT_CSV=results/miscal_sweep_G1/${EXP_NAME}/sweep.csv
mkdir -p results/miscal_sweep_G1/${EXP_NAME} logs

echo "Checkpoint: $CHECKPOINT"
echo "Output CSV: $OUTPUT_CSV"

python scripts/eval/sweep_miscal_loss.py \
    eval_data_dir=/grogu/user/harshilb/1task_new.zarr \
    val_instructions=instructions/peract/instructions.json \
    dataset=OrbitalWrist \
    bimanual=false \
    checkpoint=$CHECKPOINT \
    output_csv=$OUTPUT_CSV \
    num_samples=1024 \
    batch_size_val=64 \
    "$@"

echo "Done at: $(date)"
