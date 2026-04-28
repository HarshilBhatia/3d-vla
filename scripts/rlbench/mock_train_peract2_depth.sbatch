#!/bin/bash
#SBATCH --job-name=mock_peract2_depth
#SBATCH --partition=shubhamlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/mock_peract2_depth-%j.out
#SBATCH --error=logs/mock_peract2_depth-%j.err

# ---------------------------------------------------------------------------
# Mock sbatch for PerAct2 depth pipeline smoke test.
#
# What this does:
#   - Loads one task's zarr dataset
#   - Runs 2 batches through the depth → point-cloud preprocessor
#   - Prints tensor shapes so you can verify the depth pipeline works
#
# Adjust TASK and TRAIN_DATA_DIR before submitting.
# Submit from 3d-vla/:
#   sbatch scripts/rlbench/mock_train_peract2_depth.sbatch
# ---------------------------------------------------------------------------

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

cd /home/lzaceria/mscv/3dvla/3d-vla

source /home/lzaceria/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "GPU Info:"
nvidia-smi

TASK=${TASK:-bimanual_lift_tray}
TRAIN_DATA_DIR="Peract2_zarr/${TASK}/train.zarr"

echo "========================================================"
echo "Task           : $TASK"
echo "Train data dir : $TRAIN_DATA_DIR"
echo "========================================================"

python scripts/rlbench/mock_train_peract2_depth.py \
    --train-data-dir "$TRAIN_DATA_DIR" \
    --task           "$TASK" \
    --batch-size     2 \
    --chunk-size     1 \
    --num-batches    2 \
    --num-workers    0

echo "Job finished at: $(date)"
