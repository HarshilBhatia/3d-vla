#!/bin/bash
# SLURM: --gres=gpu:4 --mem=200G --cpus-per-task=50

# Usage:
#   bash   scripts/train/train_full/train_full.sh experiment=camtoken_deltaM_full run_log_dir=deltaM_full_fixed_medium_randnoise [overrides...]
#   sbatch scripts/train/train_full/train_full.sh experiment=camtoken_deltaM_full run_log_dir=deltaM_full_fixed_medium_randnoise [overrides...]

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

source "${REPO_ROOT}/scripts/helpers/slurm_utils.sh"
load_cluster "${CLUSTER:-$CLUSTER_NAME}"

MASTER_PORT=$((27500 + RANDOM % 1000))
ngpus=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}

main_dir=Orbital
train_epochs=7000
train_data_dir=$CLUSTER_ORB_DATA/peract_orb_new/train.zarr
eval_data_dir=$CLUSTER_ORB_DATA/peract_orb_new/val.zarr

stage_data train_data_dir
stage_data eval_data_dir

TORCHELASTIC_ERROR_FILE=/tmp/err.json WANDB_API_KEY=$WANDB_API_KEY torchrun --nproc_per_node $ngpus --master_port $MASTER_PORT \
    main.py \
    data=orbital \
    exp_log_dir=$main_dir \
    train_epochs=$train_epochs \
    train_data_dir=$train_data_dir \
    eval_data_dir=$eval_data_dir \
    batch_size=512 \
    "$@"
