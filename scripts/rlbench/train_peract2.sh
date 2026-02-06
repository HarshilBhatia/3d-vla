main_dir=Peract2

# Specify the task you want to train on
TASK=bimanual_lift_tray

DATA_PATH=$(pwd)

train_data_dir=$DATA_PATH/Peract2_zarr/$TASK/train.zarr
eval_data_dir=$DATA_PATH/Peract2_zarr/$TASK/val.zarr
train_instructions=instructions/peract2/instructions.json
val_instructions=instructions/peract2/instructions.json

dataset=Peract2_3dfront_3dwrist
num_workers=4
B=16  # Reduced from 64 to avoid OOM - you can increase if you have more GPU memory
B_val=16
chunk_size=1
memory_limit=8  # this means 8GB CPU RAM per worker per GPU,
# but it will never reach that, because these datasets are small
# reduce this if you can't allocate more than 96GB of CPU memory

# Training/testing arguments
val_freq=500
eval_only=false # this toggles eval and train
lr=1e-4
backbone_lr=1e-6  # doesn't matter when we don't finetune
lr_scheduler=constant
wd=1e-10
train_iters=30000
use_compile=false  # much faster, but sometimes unstable
use_ema=false
lv2_batch_size=1  # you can increase this and divide B equally, speed/accuracy tradeoff

# Model arguments, change (some of) these for new architectures
model_type=denoise3d
bimanual=true
keypose_only=true
pre_tokenize=true
workspace_normalizer_buffer=0.05

backbone=clip
finetune_backbone=false
finetune_text_encoder=false
fps_subsampling_factor=4

# Adaptive Trajectory-Centric Sampling (ATCS)
# Can be overridden per-job via environment variables:
#   ADAPTIVE_TRAJ_SAMPLING=true|false
#   TRAJ_SAMPLING_SIGMA=0.03
#   TRAJ_SAMPLING_BETA=1.0
adaptive_traj_sampling=${ADAPTIVE_TRAJ_SAMPLING:-true}
traj_sampling_sigma=${TRAJ_SAMPLING_SIGMA:-0.03}
traj_sampling_beta=${TRAJ_SAMPLING_BETA:-1.0}

C=120
num_attn_heads=8
num_vis_instr_attn_layers=3
num_history=3

num_shared_attn_layers=4
relative_action=false
rotation_format=quat_xyzw
denoise_timesteps=5
denoise_model=rectified_flow

# Wandb logging configuration (optional)
wandb_project=3dvla  # Change this to customize your wandb project name
wandb_name=baseline  # Leave empty to use run_log_dir as the run name, or set a custom name

run_log_dir=$model_type-$dataset-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model
checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth

ngpus=1  # we used 4

torchrun --nproc_per_node $ngpus --master_port $RANDOM \
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
    --adaptive_traj_sampling $adaptive_traj_sampling \
    --traj_sampling_sigma $traj_sampling_sigma \
    --traj_sampling_beta $traj_sampling_beta \
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
    --wandb_project $wandb_project \
    ${wandb_name:+--wandb_name $wandb_name} \
    --filter_tasks $TASK
