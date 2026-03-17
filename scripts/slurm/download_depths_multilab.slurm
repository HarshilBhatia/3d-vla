#!/bin/bash
#SBATCH --job-name=download_depths_multilab
#SBATCH --output=logs/download_depths_multilab_%j.log
#SBATCH --error=logs/download_depths_multilab_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=32
#SBATCH --account=bgkz-delta-cpu
#SBATCH --partition=cpu

source ~/.bashrc
micromamba activate gr00t

mkdir -p /u/hbhatia1/3d-vla/logs

cd /u/hbhatia1/3d-vla
python data_processing/download_multilab_depths.py \
    --selected-episodes-file /work/nvme/bgkz/droid_multilab_depths/selected_episodes.json \
    --output-dir             /work/nvme/bgkz/droid_multilab_raw \
    --num-workers            32
