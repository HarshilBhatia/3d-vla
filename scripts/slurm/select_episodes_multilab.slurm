#!/bin/bash
#SBATCH --job-name=select_episodes
#SBATCH --output=logs/select_episodes_%j.log
#SBATCH --error=logs/select_episodes_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --account=bgkz-delta-cpu
#SBATCH --partition=cpu

source ~/.bashrc
micromamba activate gr00t

mkdir -p /u/hbhatia1/3d-vla/logs
mkdir -p /work/nvme/bgkz/droid_multilab_depths

cd /u/hbhatia1/3d-vla
python data_processing/select_episodes.py \
    --dataset-path /work/nvme/bgkz/droid_raw_large_superset \
    --output-file  /work/nvme/bgkz/droid_multilab_depths/selected_episodes.json \
    --labs TRI AUTOLab IPRL REAL CLVR ILIAD IRIS \
    --max-per-lab 200
