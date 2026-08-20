#!/bin/bash
# Long 3DFA training on LIBERO-Spatial, 2xB200 devbox, uv env (no docker).
# Run on the devbox: bash scripts/train/train_libero_spatial_devbox.sh [overrides...]
#
# Env knobs:
#   BATCH_SIZE  global batch (default 12288 ≈ ~100GB/GPU on B200)
#   RUN_NAME    wandb run + checkpoint dir name
#   DATA        data config (libero_spatial | libero_spatial_traj)
#   VENV        python venv (default /root/.venv)
#   PORT        torchrun master port
set -euo pipefail

# Checkpoints land on shared NFS and are read by pods/devboxes with different
# squashed identities — make them world-readable at creation.
umask 000

cd /k8s-nfs/harsvbha/3dfa/repo

BATCH_SIZE=${BATCH_SIZE:-12288}
RUN_NAME=${RUN_NAME:-libero_spatial_base}
DATA=${DATA:-libero_spatial}
VENV=${VENV:-/root/.venv}
PORT=${PORT:-27915}

# Devbox credentials (~/.netrc) are for the FAR wandb server, not wandb.ai.
export WANDB_BASE_URL=https://far.wandb.io
export PYTORCH_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1 $VENV/bin/torchrun \
    --nproc_per_node 2 --master_port $PORT \
    main.py \
    data=$DATA \
    experiment=default \
    exp_log_dir=Libero \
    run_log_dir=$RUN_NAME \
    train_iters=100000 \
    batch_size=$BATCH_SIZE \
    preload=true \
    checkpoint=train_logs/Libero/$RUN_NAME/last.pth \
    wandb_project=libero_3dfa \
    wandb_entity=null \
    wandb_run_name=$RUN_NAME \
    "$@"
