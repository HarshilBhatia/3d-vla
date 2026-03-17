#!/bin/bash
#SBATCH --job-name=cache_depth_multilab
#SBATCH --output=logs/cache_depth_multilab_%A_%a.log
#SBATCH --error=logs/cache_depth_multilab_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --account=bgkz-delta-cpu
#SBATCH --partition=cpu
#SBATCH --array=0-31

source ~/.bashrc
micromamba activate gr00t

mkdir -p /u/hbhatia1/3d-vla/logs

cd /u/hbhatia1/3d-vla

RANK=${SLURM_ARRAY_TASK_ID}
WORLD_SIZE=${SLURM_ARRAY_TASK_COUNT}

echo "Rank $RANK / $WORLD_SIZE"

# ~387 shards / 32 workers = ~13 shards per rank
python data_processing/cache_depth_features.py \
    --backbone-cache-dir /work/nvme/bgkz/droid_multilab_cache \
    --depth-dir          /work/nvme/bgkz/droid_multilab_depths \
    --raw-dir            /work/nvme/bgkz/droid_multilab_raw \
    --rank               $RANK \
    --world-size         $WORLD_SIZE
