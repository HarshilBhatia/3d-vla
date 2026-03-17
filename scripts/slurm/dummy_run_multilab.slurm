#!/bin/bash
#SBATCH --job-name=dummy_multilab
#SBATCH --output=logs/dummy_multilab_%j.log
#SBATCH --error=logs/dummy_multilab_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --account=bgkz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1

source ~/.bashrc
micromamba activate gr00t

mkdir -p /u/hbhatia1/3d-vla/logs

cd /u/hbhatia1/3d-vla
python -m gr00t.experiment.launch_finetune \
    --base-model-path            nvidia/GR00T-N1.6-3B \
    --dataset-path               /work/nvme/bgkz/droid_raw_large_superset \
    --embodiment-tag             OXE_DROID \
    --output-dir                 /tmp/dummy_multilab_run \
    --cached-backbone-dir        /work/nvme/bgkz/droid_multilab_cache \
    --depth-dir                  /work/nvme/bgkz/droid_multilab_depths \
    --allowed-indices-file       /work/nvme/bgkz/droid_multilab_depths/selected_episodes.json \
    --shard-size                 1024 \
    --episode-sampling-rate      0.1 \
    --max-steps                  10 \
    --global-batch-size          4 \
    --dataloader-num-workers     1 \
    --no-tune-llm \
    --no-tune-visual \
    --no-tune-projector \
    --tune-diffusion-model
