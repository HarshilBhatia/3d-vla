#!/bin/bash
#SBATCH --job-name=rail_train
#SBATCH --output=logs/rail_train_%j.log
#SBATCH --error=logs/rail_train_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --account=bgkz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1

source ~/.bashrc
micromamba activate gr00t

export TRITON_CACHE_DIR=/scratch/bgkz/hbhatia1/triton_cache
export WANDB_SAVE_CODE=false

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs"

mkdir -p /u/hbhatia1/3d-vla/logs
mkdir -p /u/hbhatia1/3d-vla/outputs_rail

cd /u/hbhatia1/3d-vla

WANDB_RUN_ID=${WANDB_RUN_ID:-$(cat ./outputs_rail/.wandb_run_id 2>/dev/null || python -c "import wandb; print(wandb.util.generate_id())" | tee ./outputs_rail/.wandb_run_id)}

torchrun --nproc_per_node=$NUM_GPUS -m gr00t.experiment.launch_finetune \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path /work/nvme/bgkz/droid_raw_large_superset \
    --embodiment-tag OXE_DROID \
    --cached-backbone-dir /work/nvme/bgkz/droid_rail_cache \
    --labs RAIL \
    --output-dir /work/hdd/bgkz/hbhatia1/outputs_rail \
    --num-gpus $NUM_GPUS \
    --global-batch-size 64 \
    --max-steps 100000 \
    --shard-size 10000 \
    --dataloader-num-workers 1 \
    --save-total-limit 2 \
    --use-wandb \
    --wandb-project gr00t-droid \
    --wandb-run-name rail-cached-v1
