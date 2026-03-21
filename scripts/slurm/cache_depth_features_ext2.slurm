#!/bin/bash
#SBATCH --job-name=cache_depth_ext2
#SBATCH --output=/u/hbhatia1/3d-vla/logs/cache_depth_ext2_%A_%a.log
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

RANK=${SLURM_ARRAY_TASK_ID}
WORLD_SIZE=16

echo "Rank $RANK / $WORLD_SIZE"

# ── metadata extrinsics + ext2 ────────────────────────────────────────────────
mkdir -p /work/nvme/bgkz/droid_multilab_depth_metadata_ext2

echo "=== metadata + ext2 ==="
python data_processing/cache_depth_features.py \
    --backbone-cache-dir /work/nvme/bgkz/droid_multilab_cache_ext2 \
    --depth-dir          /work/nvme/bgkz/droid_multilab_depths \
    --raw-dir            /work/nvme/bgkz/droid_multilab_raw \
    --output-dir         /work/nvme/bgkz/droid_multilab_depth_metadata_ext2 \
    --use-ext2 \
    --rank               $RANK \
    --world-size         $WORLD_SIZE

# ── cam2cam extrinsics + ext2 ─────────────────────────────────────────────────
mkdir -p /work/nvme/bgkz/droid_multilab_depth_cam2cam_ext2

echo "=== cam2cam + ext2 ==="
python data_processing/cache_depth_features.py \
    --backbone-cache-dir /work/nvme/bgkz/droid_multilab_cache_ext2 \
    --depth-dir          /work/nvme/bgkz/droid_multilab_depths \
    --raw-dir            /work/nvme/bgkz/droid_multilab_raw \
    --cam2cam-json       /work/nvme/bgkz/droid_annotations/cam2cam_extrinsics.json \
    --output-dir         /work/nvme/bgkz/droid_multilab_depth_cam2cam_ext2 \
    --use-ext2 \
    --rank               $RANK \
    --world-size         $WORLD_SIZE
