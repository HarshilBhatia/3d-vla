#!/bin/bash
#SBATCH --job-name=dino_1task
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --partition=shubhamlong

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /home/harshilb/3d_flowmatch_actor
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

MASTER_PORT=$((27500 + RANDOM % 1000))
ngpus=$(nvidia-smi -L | wc -l)

DATASET_NAME=bimanual_lift_tray
DATA_PATH=${DATA_PATH:-/grogu/user/harshilb/datasets}

train_data_dir=$DATA_PATH/Peract2_zarr/${DATASET_NAME}/train.zarr
eval_data_dir=$DATA_PATH/Peract2_zarr/${DATASET_NAME}/val.zarr
train_instructions=instructions/peract2/instructions_${DATASET_NAME}.json
val_instructions=instructions/peract2/instructions_${DATASET_NAME}.json

experiment=default
run_log_dir=new_dino_${experiment}_${DATASET_NAME}
checkpoint=train_logs/exp/${run_log_dir}/last.pth

echo "DATASET_NAME:      $DATASET_NAME"
echo "train_data_dir:    $train_data_dir"
echo "eval_data_dir:     $eval_data_dir"
echo "experiment:        $experiment"
echo "run_log_dir:       $run_log_dir"

WANDB_API_KEY=$WANDB_API_KEY torchrun --nproc_per_node $ngpus --master_port $MASTER_PORT \
    main.py \
    data=single \
    experiment=$experiment \
    backbone=dino \
    text_backbone=clip \
    train_data_dir="$train_data_dir" \
    eval_data_dir="$eval_data_dir" \
    train_instructions="$train_instructions" \
    val_instructions="$val_instructions" \
    run_log_dir=$run_log_dir \
    checkpoint="$checkpoint"
