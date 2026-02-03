#!/bin/bash
# Default configuration for 3D FlowMatch Actor training
# These are stable defaults that rarely change
# Override these in your SLURM scripts as needed

# ============================================
# Data Configuration
# ============================================
main_dir=Peract2
DATA_PATH=$(pwd)

dataset=Peract2_3dfront_3dwrist
num_workers=4
chunk_size=1
memory_limit=8

# ============================================
# Model Architecture (rarely changed)
# ============================================
model_type=denoise3d
bimanual=true
keypose_only=true
pre_tokenize=true
workspace_normalizer_buffer=0.05

# Backbone settings
backbone=clip
finetune_backbone=false
finetune_text_encoder=false
fps_subsampling_factor=4

# Model dimensions
C=120
num_attn_heads=8
num_vis_instr_attn_layers=3
num_history=3
num_shared_attn_layers=4

# Action and rotation settings
relative_action=false
rotation_format=quat_xyzw
denoise_timesteps=5
denoise_model=rectified_flow

MASTER_PORT=$((27500 + RANDOM % 1000))

# ============================================
# Default Training Parameters
# ============================================
# These can be overridden in your SLURM script
B=${B:-64}
B_val=${B_val:-64}
lv2_batch_size=${lv2_batch_size:-1}

lr=${lr:-1e-4}
backbone_lr=${backbone_lr:-1e-6}
lr_scheduler=${lr_scheduler:-constant}
wd=${wd:-1e-10}

val_freq=${val_freq:-4000}
eval_only=${eval_only:-false}
use_compile=${use_compile:-false}
use_ema=${use_ema:-false}

# Default experiment flags (often overridden)
learn_extrinsics=${learn_extrinsics:-False}
traj_scene_rope=${traj_scene_rope:-true}
front_camera_frame=${front_camera_frame:-false}
pc_rotate_by_front_camera=${pc_rotate_by_front_camera:-false}
