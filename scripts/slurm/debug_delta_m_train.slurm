#!/bin/bash
#SBATCH --job-name=debug_delta_m_train
#SBATCH --output=logs/debug_delta_m_train_%j.log
#SBATCH --error=logs/debug_delta_m_train_%j.err
#SBATCH --time=00:30:00
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
mkdir -p /work/hdd/bgkz/hbhatia1/outputs_debug_delta_m

cd /u/hbhatia1/3d-vla

echo "Using 1 GPU on $(hostname)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=1 --master_port=$(shuf -i 29500-30000 -n 1) \
    -m gr00t.experiment.launch_finetune \
        --base-model-path            nvidia/GR00T-N1.6-3B \
        --dataset-path               /work/hdd/bgkz/droid_raw_large_superset \
        --embodiment-tag             OXE_DROID_EXT2 \
        --output-dir                 /work/hdd/bgkz/hbhatia1/outputs_debug_delta_m \
        --cached-backbone-dir        /work/nvme/bgkz/droid_multilab_cache_ext2 \
        --depth-cache-dir            /work/nvme/bgkz/droid_multilab_depth_cam2cam_ext2 \
        --allowed-indices-file       /work/nvme/bgkz/droid_multilab_depths/selected_episodes.json \
        --shard-size                 64 \
        --episode-sampling-rate      0.00035 \
        --global-batch-size          8 \
        --dataloader-num-workers     1 \
        --save-steps                 9999 \
        --save-total-limit           1 \
        --max-steps                  5 \
        --no-tune-llm \
        --no-tune-visual \
        --no-tune-projector \
        --tune-diffusion-model \
        --use-3d-rope \
        --use-delta-m \
        --num-cameras                2
