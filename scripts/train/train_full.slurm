#!/bin/bash
#SBATCH --job-name=3dfa
#SBATCH --output=logs/train_full_%j.out
#SBATCH --error=logs/train_full_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --partition=shubhamlong
#SBATCH --nodelist=grogu-2-9

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
train_data_dir=$DATA_PATH/Peract2_zarr/train.zarr
eval_data_dir=$DATA_PATH/Peract2_zarr/val.zarr
train_instructions=instructions/peract2/instructions_full.json
val_instructions=instructions/peract2/instructions_full.json

train_iters=300000

learn_extrinsics=false
traj_scene_rope=false
use_front_camera_frame=false
run_log_dir=full-LE${learn_extrinsics}-traj_scene_rope${traj_scene_rope}-front-cam-${use_front_camera_frame}
checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth

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
    learn_extrinsics=$learn_extrinsics \
    traj_scene_rope=$traj_scene_rope \
    use_front_camera_frame=$use_front_camera_frame
