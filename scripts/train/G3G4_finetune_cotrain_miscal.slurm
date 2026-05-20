#!/bin/bash
#SBATCH --job-name=G3G4_finetune_cotrain_miscal
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=all
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,9,14],grogu-3-25

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"


source "${REPO_ROOT}/scripts/helpers/slurm_utils.sh"
load_cluster "${CLUSTER:-grogu}"

MASTER_PORT=$((27500 + RANDOM % 1000))
ngpus=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}

data=orbital
main_dir=final
experiment=default
run_log_dir=3dfa_G3G4_finetune_cotrain_miscal
train_epochs=3000

train_data_dir=/grogu/user/harshilb/multi_cam/train.zarr
eval_data_dir=/grogu/user/harshilb/multi_cam/val.zarr

stage_data train_data_dir
stage_data eval_data_dir

pretrained_checkpoint=train_logs/${main_dir}/3dfa_multi_cam_G3G4/last.pth
checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth

TORCHELASTIC_ERROR_FILE=/tmp/err.json WANDB_API_KEY=$WANDB_API_KEY torchrun --nproc_per_node $ngpus --master_port $MASTER_PORT \
    main.py \
    data=$data \
    experiment=$experiment \
    exp_log_dir=$main_dir \
    run_log_dir=$run_log_dir \
    train_data_dir=$train_data_dir \
    eval_data_dir=$eval_data_dir \
    pretrained_checkpoint=$pretrained_checkpoint \
    checkpoint=$checkpoint \
    batch_size=256 \
    train_epochs=$train_epochs \
    "cotrain_miscal_group_ids=[1,2]" \
    cotrain_miscal_level=medium \
    "$@"
