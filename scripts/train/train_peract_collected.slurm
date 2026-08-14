#!/bin/bash
# Train default 3DFA on self-collected PerAct data (light_bulb_in, stack_blocks, put_money_in_safe).
# No modifications — default experiment config.
#
# Usage:
#   sbatch scripts/train/train_peract_collected.slurm

#SBATCH --job-name=train_peract
#SBATCH --partition=shubhamlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_peract_collected_%j.out
#SBATCH --error=logs/train_peract_collected_%j.err
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,9,14],grogu-3-25

set -euo pipefail

echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

source "${REPO_ROOT:-/home/harshilb/3d_flowmatch_actor}/scripts/helpers/slurm_utils.sh"
load_cluster "${CLUSTER:-grogu}"

MASTER_PORT=$((27500 + RANDOM % 1000))
ngpus=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}

train_data_dir=/grogu/datasets/hbhatia/peract_rollouts_new_zarr/train.zarr
eval_data_dir=/grogu/datasets/hbhatia/peract_rollouts_new_zarr/train.zarr
run_log_dir=peract_collected_new

checkpoint=train_logs/Peract/${run_log_dir}/last.pth

TORCHELASTIC_ERROR_FILE=/tmp/err.json WANDB_API_KEY=$WANDB_API_KEY \
torchrun --nproc_per_node $ngpus --master_port $MASTER_PORT \
    main.py \
    data=peract_collected \
    experiment=default \
    exp_log_dir=Peract \
    run_log_dir=$run_log_dir \
    train_data_dir=$train_data_dir \
    eval_data_dir=$eval_data_dir \
    bimanual=false \
    batch_size=128 \
    train_iters=50000 \
    ${checkpoint:+checkpoint=$checkpoint} \
    "$@"

echo "Done at: $(date)"
