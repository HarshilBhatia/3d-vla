#!/bin/bash
#SBATCH --job-name=cache_backbone_ext2
#SBATCH --output=/u/hbhatia1/3d-vla/logs/cache_backbone_ext2_%A_%a.log
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --account=bgkz-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA100x8
#SBATCH --gres=gpu:1

source ~/.bashrc
micromamba activate gr00t
cd /u/hbhatia1/3d-vla
mkdir -p logs
mkdir -p /work/nvme/bgkz/droid_multilab_cache_ext2

RANK=${SLURM_ARRAY_TASK_ID}
WORLD_SIZE=16

echo "Rank $RANK / $WORLD_SIZE — backbone caching (ext2 variant)"

python data_processing/cache_backbone_features.py \
    --dataset-path           /work/hdd/bgkz/droid_raw_large_superset \
    --embodiment-tag         OXE_DROID_EXT2 \
    --output-dir             /work/nvme/bgkz/droid_multilab_cache_ext2 \
    --allowed-indices-file   /work/nvme/bgkz/droid_multilab_depths/selected_episodes.json \
    --shard-size             1024 \
    --episode-sampling-rate  0.1 \
    --rank                   $RANK \
    --world-size             $WORLD_SIZE
