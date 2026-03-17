#!/bin/bash
#SBATCH --job-name=eval_3drope_with_noise
#SBATCH --output=logs/eval_3drope_with_noise_%j.log
#SBATCH --error=logs/eval_3drope_with_noise_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --account=bgkz-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --gres=gpu:1

source ~/.bashrc
micromamba activate gr00t

cd /u/hbhatia1/3d-vla
mkdir -p logs

CKPT=/work/hdd/bgkz/hbhatia1/outputs_rail_3drope_2/rail-3drope-v1/checkpoint-100000
DATASET=/work/nvme/bgkz/droid_raw_large_superset
CACHE=/work/nvme/bgkz/droid_rail_cache

# 0.05 = 5cm noise std on the 3D token positions (in meters)
NOISE_STD=1

echo "Evaluating checkpoint (noise_std=$NOISE_STD): $CKPT"

torchrun --nproc_per_node=1 -m gr00t.experiment.launch_finetune \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path $DATASET \
    --embodiment-tag OXE_DROID \
    --cached-backbone-dir $CACHE \
    --labs RAIL \
    --resume-from-checkpoint $CKPT \
    --output-dir /tmp/eval_3drope_with_noise \
    --num-gpus 1 \
    --global-batch-size 64 \
    --max-steps 50 \
    --shard-size 10000 \
    --dataloader-num-workers 1 \
    --save-total-limit 0 \
    --rope-position-noise-std $NOISE_STD \
    --eval-only

echo "Done."
