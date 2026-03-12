#!/bin/bash
#SBATCH --job-name=rail_cache
#SBATCH --output=logs/rail_cache_%A_%a.log
#SBATCH --error=logs/rail_cache_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --account=bgkz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1

RANK=${SLURM_ARRAY_TASK_ID}
WORLD_SIZE=4

source ~/.bashrc
micromamba activate gr00t

mkdir -p /u/hbhatia1/3d-vla/logs

cd /u/hbhatia1/3d-vla
python data_processing/cache_backbone_features.py \
    --dataset-path /work/nvme/bgkz/droid_raw_large_superset \
    --output-dir /work/nvme/bgkz/droid_rail_cache \
    --batch-size 128 \
    --shard-size 10000 \
    --labs RAIL \
    --rank "$RANK" \
    --world-size "$WORLD_SIZE" \
