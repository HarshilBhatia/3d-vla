#!/bin/bash
#SBATCH --job-name=analyze_extrinsics
#SBATCH --output=logs/analyze_extrinsics_%j.log
#SBATCH --error=logs/analyze_extrinsics_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --account=bgkz-delta-cpu
#SBATCH --partition=cpu

source ~/.bashrc
micromamba activate gr00t

mkdir -p /u/hbhatia1/3d-vla/logs

cd /u/hbhatia1/3d-vla

python data_processing/analyze_extrinsics.py \
    --raw-dir /work/nvme/bgkz/droid_multilab_raw \
    --output extrinsic_data.pkl \
    --plot extrinsic_plots.png \
    --workers 32
