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


# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Change to the project directory
cd /home/harshilb/3d_flowmatch_actor

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

# Load default configuration
source jobs/default_config.sh

# ============================================
# Experiment-specific overrides
# ============================================

# dataset 
# DATASET_NAME=bimanual_lift_tray
# train_data_dir=$DATA_PATH/Peract2_zarr/${DATASET_NAME}/train.zarr
# eval_data_dir=$DATA_PATH/Peract2_zarr/${DATASET_NAME}/val.zarr

# train_instructions=instructions/peract2/instructions_bimanual_lift_tray.json
# val_instructions=instructions/peract2/instructions_bimanual_lift_tray.json


# DATASET_NAME=bimanual_lift_tray,bimanual_push_box
# train_data_dir=$DATA_PATH/Peract2_zarr/${DATASET_NAME}/train.zarr
# eval_data_dir=$DATA_PATH/Peract2_zarr/${DATASET_NAME}/val.zarr

# train_instructions=instructions/peract2/instructions.json
# val_instructions=instructions/peract2/instructions.json



train_data_dir=$DATA_PATH/Peract2_zarr/train.zarr
eval_data_dir=$DATA_PATH/Peract2_zarr/val.zarr

train_instructions=instructions/peract2/instructions_full.json
val_instructions=instructions/peract2/instructions_full.json



B=64 # with 4 gpu -- 16 per gpu, so A5000s work! 
train_iters=400000

# Experiment configuration
learn_extrinsics=false
traj_scene_rope=true
predict_extrinsics=false
front_camera_frame=false

#
rope_type=normal
rope_schedule_type=none
rope_schedule_steps=$train_iters
sa_blocks_use_rope=false

run_log_dir=DroPE-full-2-rope_type-$rope_type-sa_blocks_use_rope-$sa_blocks_use_rope

checkpoint="/home/harshilb/3d_flowmatch_actor/train_logs/Peract2/DroPE-full-rope_type-normal-sa_blocks_use_rope-false/last.pth"

# /home/harshilb/3d_flowmatch_actor/train_logs/Peract2/2scene-LEFalse-traj_scene_ropetrue-front-cam-false/last.pth
# checkpoint='.pth'


echo "train_data_dir: $train_data_dir"
echo "eval_data_dir: $eval_data_dir"
echo learning extrinsics: $learn_extrinsics
echo traj_scene_rope: $traj_scene_rope
echo front_camera_frame: $front_camera_frame
echo run_log_dir: $run_log_dir



ngpus=$(nvidia-smi -L | wc -l)

WANDB_API_KEY=$WANDB_API_KEY torchrun --nproc_per_node $ngpus --master_port $MASTER_PORT\
    main.py \
    --train_data_dir $train_data_dir \
    --eval_data_dir $eval_data_dir \
    --train_instructions $train_instructions \
    --val_instructions $val_instructions \
    --dataset $dataset \
    --num_workers $num_workers \
    --batch_size $B \
    --batch_size_val $B_val \
    --chunk_size $chunk_size \
    --memory_limit $memory_limit \
    --exp_log_dir $main_dir \
    --run_log_dir ${run_log_dir} \
    --checkpoint $checkpoint \
    --val_freq $val_freq \
    --eval_only $eval_only \
    --lr $lr \
    --backbone_lr $backbone_lr \
    --lr_scheduler $lr_scheduler \
    --wd $wd \
    --train_iters $train_iters \
    --use_compile $use_compile \
    --use_ema $use_ema \
    --lv2_batch_size $lv2_batch_size \
    --model_type $model_type \
    --bimanual $bimanual \
    --keypose_only $keypose_only \
    --pre_tokenize $pre_tokenize \
    --backbone $backbone \
    --finetune_backbone $finetune_backbone \
    --finetune_text_encoder $finetune_text_encoder \
    --fps_subsampling_factor $fps_subsampling_factor \
    --embedding_dim $C \
    --num_attn_heads $num_attn_heads \
    --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
    --num_history $num_history \
    --num_shared_attn_layers $num_shared_attn_layers \
    --workspace_normalizer_buffer $workspace_normalizer_buffer \
    --relative_action $relative_action \
    --rotation_format $rotation_format \
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model \
    --use_wandb true \
    --wandb_project 3d_flowmatch_actor \
    --wandb_run_name $run_log_dir \
    --learn_extrinsics $learn_extrinsics \
    --use_front_camera_frame $front_camera_frame \
    --pc_rotate_by_front_camera $pc_rotate_by_front_camera \
    --traj_scene_rope $traj_scene_rope \
    --predict_extrinsics $predict_extrinsics \
    --rope_type $rope_type \
    --rope_schedule_type $rope_schedule_type \
    --rope_schedule_steps $rope_schedule_steps \
    --sa_blocks_use_rope $sa_blocks_use_rope