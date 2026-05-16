#!/bin/bash
#SBATCH --job-name=cotrain_mixed_miscal_dense
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=shubhamlong
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,9,14],grogu-3-25

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /home/harshilb/3d_flowmatch_actor
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

MASTER_PORT=$((27500 + RANDOM % 1000))
ngpus=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}

data=orbital
main_dir=Orbital
experiment=orb_deltaM_full
run_log_dir=cotrain_mixed_miscal_dense
train_epochs=6200

train_data_dir=/grogu/user/harshilb/dense_new/G3_G4.zarr
# eval_data_dir=/grogu/user/harshilb/multi_cam/val.zarr

source /home/harshilb/3d_flowmatch_actor/scripts/helpers/slurm_utils.sh
to_scratch train_data_dir
to_scratch eval_data_dir

checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth

TORCHELASTIC_ERROR_FILE=/tmp/err.json WANDB_API_KEY=$WANDB_API_KEY torchrun --nproc_per_node $ngpus --master_port $MASTER_PORT \
    main.py \
    data=$data \
    experiment=$experiment \
    exp_log_dir=$main_dir \
    run_log_dir=$run_log_dir \
    train_epochs=$train_epochs \
    train_data_dir=$train_data_dir \
    eval_data_dir=$train_data_dir \
    checkpoint=$checkpoint \
    batch_size=512 \
    keypose_only=false \
    "cotrain_miscal_group_ids=[4]" \
    cotrain_miscal_level=medium \
    "$@"
