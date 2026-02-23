#!/bin/bash
#SBATCH --job-name=DroPE
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

# ============================================
# Overrides only
# ============================================

train_iters=400000
batch_size=64

run_log_dir=DroPE-full-2-rope_type-normal-sa_blocks_use_rope-false
checkpoint=/home/harshilb/3d_flowmatch_actor/train_logs/Peract2/DroPE-full-rope_type-normal-sa_blocks_use_rope-false/last.pth

predict_extrinsics=false
use_front_camera_frame=false
rope_schedule_type=none
rope_schedule_steps=$train_iters
sa_blocks_use_rope=false

echo "train_data_dir: $train_data_dir"
echo "eval_data_dir: $eval_data_dir"
echo "run_log_dir: $run_log_dir"

ngpus=$(nvidia-smi -L | wc -l)

WANDB_API_KEY=$WANDB_API_KEY torchrun --nproc_per_node $ngpus --master_port $MASTER_PORT \
    main.py \
    dataset=Peract2_3dfront_3dwrist \
    train_data_dir="$train_data_dir" \
    eval_data_dir="$eval_data_dir" \
    train_instructions="$train_instructions" \
    val_instructions="$val_instructions" \
    exp_log_dir=$main_dir \
    run_log_dir="${run_log_dir}" \
    wandb_run_name="$run_log_dir" \
    checkpoint="$checkpoint" \
    train_iters=$train_iters \
    batch_size=$batch_size \
    batch_size_val=$batch_size \
    predict_extrinsics=$predict_extrinsics \
    use_front_camera_frame=$use_front_camera_frame \
    rope_schedule_type=$rope_schedule_type \
    rope_schedule_steps=$rope_schedule_steps \
    sa_blocks_use_rope=$sa_blocks_use_rope
