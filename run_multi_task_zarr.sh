#!/bin/bash
set -e
# Training launcher. Override only what you change; rest from config/config.yaml.
source jobs/default_config.sh

# ============================================
# Overrides only
# ============================================
DATASET_NAME=bimanual_lift_tray
train_data_dir=$DATA_PATH/Peract2_zarr/${DATASET_NAME}/train.zarr
eval_data_dir=$DATA_PATH/Peract2_zarr/${DATASET_NAME}/val.zarr
train_instructions=instructions/peract2/instructions_bimanual_lift_tray.json
val_instructions=instructions/peract2/instructions_bimanual_lift_tray.json

train_iters=45000
batch_size=16

run_log_dir=Full-ComRoPE-normal-front_cam-true-cam_token-true-traj_scene_rope-true
checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth

use_front_camera_frame=true
predict_extrinsics=true
extrinsics_prediction_mode=delta_m
use_com_rope=false
com_rope_block_size=3
com_rope_num_axes=3
com_rope_init_std=0.02
rope_schedule_type=linear
rope_schedule_steps=$train_iters

echo "train_data_dir: $train_data_dir"
echo "eval_data_dir: $eval_data_dir"
echo "run_log_dir: $run_log_dir"

ngpus=$(nvidia-smi -L | wc -l)

WANDB_API_KEY=$WANDB_API_KEY torchrun --nproc_per_node $ngpus --master_port $RANDOM \
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
    use_front_camera_frame=$use_front_camera_frame \
    predict_extrinsics=$predict_extrinsics \
    extrinsics_prediction_mode=$extrinsics_prediction_mode \
    use_com_rope=$use_com_rope \
    com_rope_block_size=$com_rope_block_size \
    com_rope_num_axes=$com_rope_num_axes \
    com_rope_init_std=$com_rope_init_std \
    rope_schedule_type=$rope_schedule_type \
    rope_schedule_steps=$rope_schedule_steps
