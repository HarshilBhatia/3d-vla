main_dir=Orbital

train_data_dir=/grogu/user/harshilb/orbital_train.zarr
eval_data_dir=/grogu/user/harshilb/orbital_train.zarr

train_instructions=instructions/peract/instructions.json
val_instructions=instructions/peract/instructions.json

dataset=Peract2_3dfront_3dwrist
num_workers=4
B=32
B_val=32
chunk_size=1
memory_limit=8

# Training/testing arguments
eval_only=false
lr=1e-4
backbone_lr=1e-6
lr_scheduler=constant
wd=1e-10
train_iters=400000
use_compile=false
use_ema=false
lv2_batch_size=1

# Model arguments
model_type=denoise3d
bimanual=false
keypose_only=true
pre_tokenize=true
workspace_normalizer_buffer=0.04

backbone=clip
finetune_backbone=false
finetune_text_encoder=false
fps_subsampling_factor=4

C=120
num_attn_heads=8
num_vis_instr_attn_layers=2
num_history=3

num_shared_attn_layers=4
relative_action=false
rotation_format=quat_xyzw
denoise_timesteps=5
denoise_model=rectified_flow

# deltaM full extrinsics prediction
learn_extrinsics=false
predict_extrinsics=true
extrinsics_prediction_mode=delta_m_full
dynamic_rope_from_camtoken=true
use_front_camera_frame=false
pc_rotate_by_front_camera=false

traj_scene_rope=true

rope_type=normal
rope_schedule_type=linear
rope_schedule_steps=$train_iters

# Miscalibration noise
miscalibration_noise_level=large

run_log_dir="miscal_large_deltaM_full"
checkpoint=train_logs/${main_dir}/${run_log_dir}/interm340000.pth

ngpus=$(nvidia-smi -L | wc -l)

NCCL_DEBUG=WARN torchrun --nproc_per_node $ngpus --master_port $((27500 + RANDOM % 5000)) \
    main.py \
    train_data_dir=$train_data_dir \
    eval_data_dir=$eval_data_dir \
    train_instructions=$train_instructions \
    val_instructions=$val_instructions \
    dataset=$dataset \
    num_workers=$num_workers \
    batch_size=$B \
    batch_size_val=$B_val \
    chunk_size=$chunk_size \
    memory_limit=$memory_limit \
    exp_log_dir=$main_dir \
    run_log_dir="${run_log_dir}" \
    checkpoint="$checkpoint" \
    eval_only=$eval_only \
    lr=$lr \
    backbone_lr=$backbone_lr \
    lr_scheduler=$lr_scheduler \
    wd=$wd \
    train_iters=$train_iters \
    use_compile=$use_compile \
    use_ema=$use_ema \
    lv2_batch_size=$lv2_batch_size \
    model_type=$model_type \
    bimanual=$bimanual \
    keypose_only=$keypose_only \
    pre_tokenize=$pre_tokenize \
    backbone=$backbone \
    finetune_backbone=$finetune_backbone \
    finetune_text_encoder=$finetune_text_encoder \
    fps_subsampling_factor=$fps_subsampling_factor \
    embedding_dim=$C \
    num_attn_heads=$num_attn_heads \
    num_vis_instr_attn_layers=$num_vis_instr_attn_layers \
    num_history=$num_history \
    num_shared_attn_layers=$num_shared_attn_layers \
    workspace_normalizer_buffer=$workspace_normalizer_buffer \
    relative_action=$relative_action \
    rotation_format=$rotation_format \
    denoise_timesteps=$denoise_timesteps \
    denoise_model=$denoise_model \
    use_wandb=true \
    wandb_project=3d_flowmatch_actor \
    wandb_run_name="$run_log_dir" \
    learn_extrinsics=$learn_extrinsics \
    use_front_camera_frame=$use_front_camera_frame \
    pc_rotate_by_front_camera=$pc_rotate_by_front_camera \
    traj_scene_rope=$traj_scene_rope \
    predict_extrinsics=$predict_extrinsics \
    extrinsics_prediction_mode=$extrinsics_prediction_mode \
    dynamic_rope_from_camtoken=$dynamic_rope_from_camtoken \
    rope_type=$rope_type \
    rope_schedule_type=$rope_schedule_type \
    rope_schedule_steps=$rope_schedule_steps \
    miscalibration_noise_level=$miscalibration_noise_level
