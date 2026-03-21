#!/bin/bash
#SBATCH --job-name=plot_depth_ranges
#SBATCH --output=logs/plot_depth_ranges_%j.log
#SBATCH --error=logs/plot_depth_ranges_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=64
#SBATCH --account=bgkz-delta-cpu
#SBATCH --partition=cpu

source ~/.bashrc
micromamba activate gr00t

mkdir -p /u/hbhatia1/3d-vla/logs

cd /u/hbhatia1/3d-vla

python data_processing/plot_depth_ranges.py \
    --depth-dir  /work/nvme/bgkz/droid_multilab_depths \
    --output     /work/nvme/bgkz/droid_multilab_depths/depth_ranges.png \
    --workers    64 \
    --frames-per-episode 8
