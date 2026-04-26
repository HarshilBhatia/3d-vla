#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TASK="open_drawer"
CHECKPOINT="${REPO_ROOT}/train_logs/Peract/peract_collected/last.pth"
OUTPUT_FILE="${REPO_ROOT}/eval_logs/Peract/peract_collected/results_${TASK}.json"

mkdir -p "$(dirname "$OUTPUT_FILE")" logs

xvfb-run -a python "${REPO_ROOT}/online_evaluation_rlbench/evaluate_policy.py" \
    dataset=PeractCollected \
    data_dir=/grogu/user/harshilb/peract_rollouts \
    val_instructions=instructions/peract/instructions.json \
    "image_size='128,128'" \
    bimanual=false \
    model_type=denoise3d \
    backbone=clip \
    finetune_backbone=false \
    finetune_text_encoder=false \
    fps_subsampling_factor=5 \
    embedding_dim=120 \
    num_attn_heads=8 \
    num_vis_instr_attn_layers=3 \
    num_shared_attn_layers=4 \
    relative_action=false \
    rotation_format=quat_xyzw \
    denoise_timesteps=5 \
    denoise_model=rectified_flow \
    learn_extrinsics=false \
    predict_extrinsics=false \
    dynamic_rope_from_camtoken=true \
    traj_scene_rope=true \
    rope_type=normal \
    sa_blocks_use_rope=true \
    max_steps=25 \
    prediction_len=1 \
    num_history=3 \
    max_tries=3 \
    headless=true \
    collision_checking=false \
    seed=0 \
    checkpoint=$CHECKPOINT \
    output_file=$OUTPUT_FILE \
    task=$TASK
