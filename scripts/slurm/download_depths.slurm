#!/bin/bash
#SBATCH --job-name=download_depths
#SBATCH --output=logs/download_depths_%j.log
#SBATCH --error=logs/download_depths_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=32
#SBATCH --account=bgkz-delta-cpu
#SBATCH --partition=cpu

source ~/.bashrc
micromamba activate gr00t

mkdir -p /u/hbhatia1/3d-vla/logs

cd /u/hbhatia1/3d-vla
python data_processing/download_rail_depths.py \
    --output-dir /work/nvme/bgkz/droid_rail_raw \
    --num-workers 32
