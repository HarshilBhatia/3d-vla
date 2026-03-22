#!/bin/bash
#SBATCH --job-name=multilab_3d_action_deltaM
#SBATCH --output=logs/multilab_3d_action_deltaM_%j.log
#SBATCH --error=logs/multilab_3d_action_deltaM_%j.err
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --account=bgkz-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA100x8
#SBATCH --gres=gpu:2

source ~/.bashrc
micromamba activate gr00t

mkdir -p /u/hbhatia1/3d-vla/logs
mkdir -p /work/hdd/bgkz/hbhatia1/multilab_3d_action_deltaM

cd /u/hbhatia1/3d-vla

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Using $NUM_GPUS GPUs on $(hostname)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WANDB_RUN_ID=${WANDB_RUN_ID:-$(cat /work/hdd/bgkz/hbhatia1/multilab_3d_action_deltaM/.wandb_run_id 2>/dev/null || python -c "import wandb; print(wandb.util.generate_id())" | tee /work/hdd/bgkz/hbhatia1/multilab_3d_action_deltaM/.wandb_run_id)}

MASTER_PORT=$(shuf -i 29500-30000 -n 1)
torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    -m gr00t.experiment.launch_finetune \
        --base-model-path            nvidia/GR00T-N1.6-DROID \
        --dataset-path               /work/hdd/bgkz/droid_raw_large_superset \
        --embodiment-tag             OXE_DROID_EXT2 \
        --output-dir                 /work/hdd/bgkz/hbhatia1/multilab_3d_action_deltaM \
        --cached-backbone-dir        /work/nvme/bgkz/droid_multilab_cache_ext2 \
        --depth-cache-dir            /work/nvme/bgkz/droid_multilab_depth_cam2cam_ext2 \
        --allowed-indices-file       /work/nvme/bgkz/droid_multilab_depths/selected_episodes.json \
        --shard-size                 1024 \
        --episode-sampling-rate      0.1 \
        --global-batch-size          64 \
        --max-steps                  100000 \
        --dataloader-num-workers     2 \
        --save-steps                 1000 \
        --save-total-limit           2 \
        --no-tune-llm \
        --no-tune-visual \
        --no-tune-projector \
        --tune-diffusion-model \
        --use-wandb \
        --wandb-project              gr00t-droid \
        --wandb-run-name             multi-3d-action-deltaM \
        --use-3d-rope \
        --use-action-delta-m
