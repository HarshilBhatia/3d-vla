#!/bin/bash
# SLURM: --gres=gpu:4 --mem=96G --cpus-per-task=20

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

source "${REPO_ROOT}/scripts/helpers/slurm_utils.sh"
load_cluster "${CLUSTER:-babel}"

train_data_dir=$CLUSTER_ORB_DATA/peract_orb_new/train.zarr
eval_data_dir=$CLUSTER_ORB_DATA/peract_orb_new/val.zarr

stage_data train_data_dir
stage_data eval_data_dir

MASTER_PORT=$((27500 + RANDOM % 1000))
ngpus=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}

data=orbital
main_dir=Orbital
experiment=default
run_log_dir=multicam_default_miscal
train_epochs=3000
checkpoint="${CHECKPOINT:-}"

WANDB_API_KEY=$WANDB_API_KEY torchrun --nproc_per_node $ngpus --master_port $MASTER_PORT \
    main.py \
    data=$data \
    experiment=$experiment \
    exp_log_dir=$main_dir \
    run_log_dir=$run_log_dir \
    train_epochs=$train_epochs \
    ${CLUSTER_LOG_DIR:+base_log_dir=$CLUSTER_LOG_DIR} \
    train_data_dir=$train_data_dir \
    eval_data_dir=$eval_data_dir \
    batch_size=256 \
    orbital_miscal_noise_level=medium \
    miscal_max_angle_deg=3.0 \
    miscal_max_translation_m=0.01 \
    ${checkpoint:+checkpoint=$checkpoint} \
    "$@"
