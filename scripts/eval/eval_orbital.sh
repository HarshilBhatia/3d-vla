#!/usr/bin/env bash
# Evaluate a trained orbital model on RLBench.
# Usage:
#   xvfb-run -a bash scripts/rlbench/eval_orbital.sh \
#       task=close_jar camera_group=G1 miscalibration_noise_level=small \
#       checkpoint=train_logs/Orbital/miscal_small_deltaM_full/last.pth
#
# All arguments are passed as Hydra overrides to evaluate_policy.py.

main_dir=Orbital

eval_data_dir=/grogu/user/harshilb/orbital_train.zarr

val_instructions=instructions/peract/instructions.json

dataset=OrbitalWrist
cameras_file=instructions/orbital_cameras_grouped.json
task_group_mapping_file=instructions/task_group_mapping.json
fov_deg=60.0
miscalibration_noise_level=null

image_size="256,256"
max_steps=25
prediction_len=1
num_history=3
max_tries=1

# Model arguments (must match checkpoint)
model_type=denoise3d
bimanual=false
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

learn_extrinsics=false
predict_extrinsics=true
extrinsics_prediction_mode=delta_m_full
dynamic_rope_from_camtoken=true
use_front_camera_frame=false
pc_rotate_by_front_camera=false

traj_scene_rope=true
rope_type=normal
use_com_rope=false
com_rope_block_size=0
com_rope_num_axes=0
com_rope_init_std=0.0

sa_blocks_use_rope=false

run_log_dir="miscal_small_deltaM_full"
checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth
output_file=eval_logs/${main_dir}/${run_log_dir}/results.json

task=close_jar
headless=true
collision_checking=false
seed=0

python online_evaluation_rlbench/evaluate_policy.py \
    val_instructions=$val_instructions \
    eval_data_dir=$eval_data_dir \
    dataset=$dataset \
    cameras_file=$cameras_file \
    task_group_mapping_file=$task_group_mapping_file \
    fov_deg=$fov_deg \
    miscalibration_noise_level=$miscalibration_noise_level \
    image_size=$image_size \
    max_steps=$max_steps \
    prediction_len=$prediction_len \
    num_history=$num_history \
    max_tries=$max_tries \
    model_type=$model_type \
    bimanual=$bimanual \
    backbone=$backbone \
    finetune_backbone=$finetune_backbone \
    finetune_text_encoder=$finetune_text_encoder \
    fps_subsampling_factor=$fps_subsampling_factor \
    embedding_dim=$C \
    num_attn_heads=$num_attn_heads \
    num_vis_instr_attn_layers=$num_vis_instr_attn_layers \
    num_shared_attn_layers=$num_shared_attn_layers \
    relative_action=$relative_action \
    rotation_format=$rotation_format \
    denoise_timesteps=$denoise_timesteps \
    denoise_model=$denoise_model \
    learn_extrinsics=$learn_extrinsics \
    predict_extrinsics=$predict_extrinsics \
    extrinsics_prediction_mode=$extrinsics_prediction_mode \
    dynamic_rope_from_camtoken=$dynamic_rope_from_camtoken \
    use_front_camera_frame=$use_front_camera_frame \
    pc_rotate_by_front_camera=$pc_rotate_by_front_camera \
    traj_scene_rope=$traj_scene_rope \
    rope_type=$rope_type \
    use_com_rope=$use_com_rope \
    com_rope_block_size=$com_rope_block_size \
    com_rope_num_axes=$com_rope_num_axes \
    com_rope_init_std=$com_rope_init_std \
    sa_blocks_use_rope=$sa_blocks_use_rope \
    checkpoint=$checkpoint \
    output_file=$output_file \
    task=$task \
    headless=$headless \
    collision_checking=$collision_checking \
    seed=$seed \
    "$@"
