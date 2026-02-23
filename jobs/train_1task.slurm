#!/bin/bash
#SBATCH --job-name=inter-3dfa
#SBATCH --output=logs/train_peract2_%j.out
#SBATCH --error=logs/train_peract2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --partition=shubhamlong
#SBATCH --nodelist=grogu-0-14

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /home/harshilb/3d_flowmatch_actor
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

source jobs/default_config.sh

DATASET_NAME=bimanual_lift_tray
train_data_dir=$DATA_PATH/Peract2_zarr/${DATASET_NAME}/train.zarr
eval_data_dir=$DATA_PATH/Peract2_zarr/${DATASET_NAME}/val.zarr
train_instructions=instructions/peract2/instructions_bimanual_lift_tray.json
val_instructions=instructions/peract2/instructions_bimanual_lift_tray.json
checkpoint=train_logs/${main_dir}/1scene_3dfa-front_cam-false-cam_token-false/last.pth

echo "train_data_dir: $train_data_dir"
echo "eval_data_dir: $eval_data_dir"
echo "experiment: one_task"

ngpus=$(nvidia-smi -L | wc -l)

WANDB_API_KEY=$WANDB_API_KEY torchrun --nproc_per_node $ngpus --master_port $MASTER_PORT \
    main.py \
    dataset=peract2 \
    experiment=one_task \
    train_data_dir="$train_data_dir" \
    eval_data_dir="$eval_data_dir" \
    train_instructions="$train_instructions" \
    val_instructions="$val_instructions" \
    exp_log_dir=$main_dir \
    checkpoint="$checkpoint"
