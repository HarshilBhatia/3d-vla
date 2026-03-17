#!/bin/bash
#SBATCH --job-name=check_depth_multilab
#SBATCH --output=logs/check_depth_multilab_%j.log
#SBATCH --error=logs/check_depth_multilab_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --account=bgkz-delta-cpu
#SBATCH --partition=cpu

source ~/.bashrc
micromamba activate gr00t

mkdir -p /u/hbhatia1/3d-vla/logs

cd /u/hbhatia1/3d-vla
python data_processing/check_depth_completeness.py \
    --depth-dir  /work/nvme/bgkz/droid_multilab_depths \
    --raw-dir    /work/nvme/bgkz/droid_multilab_raw \
    --serial-map /work/nvme/bgkz/droid_multilab_depths/serial_map.json
