#!/bin/bash
#SBATCH --job-name=cache_depth_campos
#SBATCH --output=/u/hbhatia1/3d-vla/logs/cache_depth_campos_%A_%a.log
#SBATCH --array=0-15
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
mkdir -p /work/nvme/bgkz/droid_multilab_depth_cam2cam_campos_ext2

RANK=${SLURM_ARRAY_TASK_ID}
WORLD_SIZE=16

echo "Rank $RANK / $WORLD_SIZE — cam2cam + ext2 + camera_positions_3d"

python data_processing/cache_depth_features.py \
    --backbone-cache-dir /work/nvme/bgkz/droid_multilab_cache_ext2 \
    --depth-dir          /work/nvme/bgkz/droid_multilab_depths \
    --raw-dir            /work/nvme/bgkz/droid_multilab_raw \
    --cam2cam-json       /work/nvme/bgkz/droid_annotations/cam2cam_extrinsics.json \
    --output-dir         /work/nvme/bgkz/droid_multilab_depth_cam2cam_campos_ext2 \
    --use-ext2 \
    --rank               $RANK \
    --world-size         $WORLD_SIZE
