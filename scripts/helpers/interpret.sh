main_dir=Peract2

DATA_PATH=$(pwd)

train_data_dir=$DATA_PATH/Peract2_zarr/bimanual_lift_tray/train.zarr
eval_data_dir=$DATA_PATH/Peract2_zarr/bimanual_lift_tray/val.zarr

# train_data_dir=$DATA_PATH/Peract2_zarr/train.zarr
# eval_data_dir=$DATA_PATH/Peract2_zarr/val.zarr

train_instructions=instructions/peract2/instructions_bimanual_lift_tray.json
val_instructions=instructions/peract2/instructions_bimanual_lift_tray.json


dataset=Peract2_3dfront_3dwrist
num_workers=16
B=16  # we used 64 but you can use as low as 16 without much performance drop - it's much faster
B_val=64
chunk_size=1
memory_limit=8 

# Training/testing arguments
val_freq=4000
eval_only=True # this toggles eval and train
lr=1e-4
backbone_lr=1e-6  # doesn't matter when we don't finetune
lr_scheduler=constant
wd=1e-10
# train_iters=300000 

train_iters=45000
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

C=120
num_attn_heads=8
num_vis_instr_attn_layers=3
num_history=3

num_shared_attn_layers=4
relative_action=false
rotation_format=quat_xyzw
denoise_timesteps=5
denoise_model=rectified_flow


# Model arguments for learning extrinsics and predicting extrinsics
learn_extrinsics=False
predict_extrinsics=True
use_front_camera_frame=false
traj_scene_rope=true
rope_type=stopgrad
rope_schedule_type=linear
rope_schedule_steps=$train_iters


# checkpoint='/home/harshilb/3d_flowmatch_actor/train_logs/Peract2/1task-cam_token_extrinsics-traj_scene_ropetrue-front-cam-true/best.pth'
# run_log_dir=analysis/cam_token_front_cam/



# checkpoint='/home/harshilb/3d_flowmatch_actor/train_logs/Peract2/1task-cam_token_extrinsics-traj_scene_ropetrue-front-cam-false/best.pth'
# run_log_dir=analysis/cam_token_front_cam_false/

# checkpoint='/home/harshilb/3d_flowmatch_actor/train_logs/Peract2/1scene_RoPEADAM-front_cam--cam_token-true-traj_scene_rope-true/best.pth'
# run_log_dir=analysis/scene_rope_adam_front_cam/


# checkpoint='/home/harshilb/3d_flowmatch_actor/train_logs/Peract2/denoise3d-Peract2_3dfront_3dwrist-C120-B64-lr1e-4-constant-H3-rectified_flow/best.pth'
# # checkpoint='.pth'
# run_log_dir=analysis/3dfa-single/

checkpoint=/home/harshilb/3d_flowmatch_actor/train_logs/Peract2/1scene_RoPEstopgrad_schedule_linear-front_cam--cam_token-true-traj_scene_rope-true/best.pth
run_log_dir=analysis/scene_rope_stopgrad_linear_front_cam/

# ngpus=2
ngpus=1
#$(nvidia-smi -L | wc -l)


torchrun --nproc_per_node $ngpus --master_port $RANDOM\
    analyse_qk.py \
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
    --use_wandb false \
    --wandb_project 3d_flowmatch_actor \
    --wandb_run_name $run_log_dir \
    --learn_extrinsics $learn_extrinsics \
    --use_front_camera_frame $use_front_camera_frame \
    --traj_scene_rope $traj_scene_rope \
    --predict_extrinsics $predict_extrinsics \
    --rope_type $rope_type \
    --rope_schedule_type $rope_schedule_type \
    --rope_schedule_steps $rope_schedule_steps
