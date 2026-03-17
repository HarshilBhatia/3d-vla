#!/bin/bash
#SBATCH --job-name=frame_index_multilab
#SBATCH --output=logs/frame_index_multilab_%j.log
#SBATCH --error=logs/frame_index_multilab_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --account=bgkz-delta-cpu
#SBATCH --partition=cpu

source ~/.bashrc
micromamba activate gr00t

mkdir -p /u/hbhatia1/3d-vla/logs

cd /u/hbhatia1/3d-vla
python data_processing/build_episode_frame_index.py \
    --dataset-path          /work/nvme/bgkz/droid_raw_large_superset \
    --raw-dir               /work/nvme/bgkz/droid_multilab_raw \
    --output-dir            /work/nvme/bgkz/droid_multilab_depths \
    --allowed-indices-file  /work/nvme/bgkz/droid_multilab_depths/selected_episodes.json \
    --shard-size            1024 \
    --episode-sampling-rate 0.1
