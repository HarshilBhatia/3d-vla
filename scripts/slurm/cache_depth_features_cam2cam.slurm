#!/bin/bash
#SBATCH --job-name=cache_depth_cam2cam
#SBATCH --output=/u/hbhatia1/3d-vla/logs/cache_depth_cam2cam_%A_%a.log
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=bgkz-delta-cpu
#SBATCH --partition=cpu

source ~/.bashrc
micromamba activate gr00t
cd /u/hbhatia1/3d-vla
mkdir -p logs
mkdir -p /work/nvme/bgkz/droid_multilab_depth_cam2cam

RANK=${SLURM_ARRAY_TASK_ID}
WORLD_SIZE=4

echo "Rank $RANK / $WORLD_SIZE — cam2cam depth caching"

python data_processing/cache_depth_features.py \
    --backbone-cache-dir /work/nvme/bgkz/droid_multilab_cache \
    --depth-dir          /work/nvme/bgkz/droid_multilab_depths \
    --raw-dir            /work/nvme/bgkz/droid_multilab_raw \
    --cam2cam-json       /work/nvme/bgkz/droid_annotations/cam2cam_extrinsics.json \
    --output-dir         /work/nvme/bgkz/droid_multilab_depth_cam2cam \
    --rank               $RANK \
    --world-size         $WORLD_SIZE
