#!/bin/bash
#SBATCH --job-name=cache_3d_pos
#SBATCH --output=logs/cache_3d_pos_%A_%a.log
#SBATCH --error=logs/cache_3d_pos_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --account=bgkz-delta-cpu
#SBATCH --partition=cpu
# CPU-only job: no GPU needed (depth unprojection is pure numpy/torch)
# Array: one task per shard (37 shards for RAIL, 0-indexed)
# sbatch --array=0-36 scripts/slurm/cache_3d_positions.slurm

RANK=${SLURM_ARRAY_TASK_ID}
WORLD_SIZE=37

source ~/.bashrc
micromamba activate gr00t

mkdir -p /u/hbhatia1/3d-vla/logs

cd /u/hbhatia1/3d-vla
python data_processing/cache_3d_positions.py \
    --dataset-path /work/nvme/bgkz/droid_raw_large_superset \
    --cache-dir /work/nvme/bgkz/droid_rail_cache \
    --pos-cache-dir /work/nvme/bgkz/droid_rail_pos_cache \
    --depth-dir /work/nvme/bgkz/droid_rail_raw/depths \
    --intrinsics-json /work/nvme/bgkz/droid_rail_raw/intrinsics.json \
    --extrinsics-json /work/nvme/bgkz/droid_rail_raw/extrinsics.json \
    --shard-size 10000 \
    --episode-sampling-rate 0.1 \
    --labs RAIL \
    --rank "$RANK" \
    --world-size "$WORLD_SIZE"
