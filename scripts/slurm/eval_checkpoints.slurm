#!/bin/bash
#SBATCH --job-name=eval_ckpts
#SBATCH --output=logs/eval_ckpts_%j.log
#SBATCH --error=logs/eval_ckpts_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --account=bgkz-delta-gpu
#SBATCH --partition=gpuH100x8
#SBATCH --gres=gpu:2

source ~/.bashrc
micromamba activate gr00t

cd /u/hbhatia1/3d-vla
mkdir -p logs

CKPT_BASE=/work/hdd/bgkz/hbhatia1/outputs_rail/rail-cached-v1
DATASET=/work/nvme/bgkz/droid_raw_large_superset
CACHE=/work/nvme/bgkz/droid_rail_cache

for CKPT in checkpoint-72000 checkpoint-73000; do
    echo "========================================"
    echo "Evaluating $CKPT"
    echo "========================================"

    torchrun --nproc_per_node=2 -m gr00t.experiment.launch_finetune \
        --base-model-path nvidia/GR00T-N1.6-3B \
        --dataset-path $DATASET \
        --embodiment-tag OXE_DROID \
        --cached-backbone-dir $CACHE \
        --labs RAIL \
        --resume-from-checkpoint $CKPT_BASE/$CKPT \
        --output-dir /tmp/eval_$CKPT \
        --num-gpus 2 \
        --global-batch-size 64 \
        --max-steps 50 \
        --shard-size 10000 \
        --dataloader-num-workers 1 \
        --save-total-limit 0 \
        --eval-only

    echo "Done with $CKPT"
done
