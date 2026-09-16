#!/bin/bash
# SLURM: --gres=gpu:2 --mem=96G --cpus-per-task=24
#
# Short, real-data DDP benchmark for the cache-aware Video-DeltaM loader.
# The benchmark logger writes data_ms separately for every rank, so the logs
# reveal loader imbalance as well as average throughput.

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

source "${REPO_ROOT}/scripts/helpers/slurm_utils.sh"
load_cluster "${CLUSTER:-$CLUSTER_NAME}"

# The cluster profiles can point at an older shared checkout.  Benchmark the
# submitted worktree so the newly added sampler is definitely what is measured.
cd "$REPO_ROOT"

train_data_dir=$CLUSTER_DATA_ROOT/multi_cam/train.zarr
eval_data_dir=$CLUSTER_DATA_ROOT/multi_cam/val.zarr

stage_data train_data_dir
stage_data eval_data_dir

MASTER_PORT=$((27500 + RANDOM % 1000))
ngpus=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}

# Keep this short enough for an interactive measurement; the purpose is loader
# wait-time, not production-scale model throughput.

WANDB_MODE=disabled torchrun --nproc_per_node "$ngpus" --master_port "$MASTER_PORT" \
    main.py \
    data=orbital \
    experiment=video_deltaM \
    miscal=orbital_fixed_medium_randnoise \
    exp_log_dir=Orbital \
    run_log_dir="benchmark_video_deltam_loader_${SLURM_JOB_ID}" \
    train_data_dir="$train_data_dir" \
    eval_data_dir="$eval_data_dir" \
    batch_size=32 \
    visual_num_history=5 \
    train_iters=60 \
    val_interval_steps=1000000 \
    ckpt_interval_steps=1000000 \
    interm_ckpt_interval_steps=1000000 \
    use_wandb=false \
    use_compile=false \
    benchmark=true \
    benchmark_warmup_steps=20 \
    benchmark_log_freq=20 \
    benchmark_profile_steps=0 \
    video_deltam_cache_batches=4 \
    video_deltam_cache_span=8 \
    ${CLUSTER_LOG_DIR:+base_log_dir=$CLUSTER_LOG_DIR} \
    "$@"
