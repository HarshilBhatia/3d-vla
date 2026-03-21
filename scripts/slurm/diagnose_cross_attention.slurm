#!/bin/bash
#SBATCH --job-name=diagnose_xattn
#SBATCH --output=logs/diagnose_xattn_%j.log
#SBATCH --error=logs/diagnose_xattn_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --account=bgkz-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA100x8
#SBATCH --gres=gpu:1

source ~/.bashrc
micromamba activate gr00t

mkdir -p /u/hbhatia1/3d-vla/logs
cd /u/hbhatia1/3d-vla

python data_processing/diagnose_cross_attention.py \
    --backbone-cache-dir /work/nvme/bgkz/droid_multilab_cache_ext2 \
    --depth-cache-dir    /work/nvme/bgkz/droid_multilab_depth_cam2cam_ext2 \
    --n-batches  50 \
    --batch-size  8 \
    --checkpoints \
        "groot_base:nvidia/GR00T-N1.6-3B" \
        "groot_droid:nvidia/GR00T-N1.6-DROID" \
        "my_baseline:/work/hdd/bgkz/hbhatia1/multilab_baseline_ext2/multilab-baseline-ext2" \
        "my_3d:/work/hdd/bgkz/hbhatia1/multilab_3d_cam2cam/multilab-3d-cam2cam-ext2"
