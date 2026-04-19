#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# online_eval.slurm
#
# Slurm array job: evaluate all 18 PerAct tasks in parallel (one job per task).
# Each job needs 1 GPU for the policy + xvfb for headless CoppeliaSim.
#
# Usage:
#   sbatch --export=CHECKPOINT=train_logs/Orbital/my_run/last.pth,RUN_LOG_DIR=my_run \
#          scripts/eval/online_eval.slurm
#
# Smoke test a single task (index 1 = close_jar):
#   sbatch --array=1 --export=CHECKPOINT=...,RUN_LOG_DIR=... scripts/eval/online_eval.slurm
#
# Resubmit failed indices:
#   sbatch --array=3,7,12 --export=... scripts/eval/online_eval.slurm
#
# Task index mapping (0-based):
#   0  place_cups                    9  put_groceries_in_cupboard
#   1  close_jar                    10  put_item_in_drawer
#   2  insert_onto_square_peg       11  put_money_in_safe
#   3  light_bulb_in                12  reach_and_drag
#   4  meat_off_grill               13  slide_block_to_color_target
#   5  open_drawer                  14  stack_blocks
#   6  place_shape_in_shape_sorter  15  stack_cups
#   7  place_wine_at_rack_location  16  sweep_to_dustpan_of_size
#   8  push_buttons                 17  turn_tap
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=online_eval
#SBATCH --partition=shubhamlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/online_eval_%A_%a.out
#SBATCH --error=logs/online_eval_%A_%a.err

echo "Job started at: $(date)"
echo "Node: $(hostname)  Array index: $SLURM_ARRAY_TASK_ID"

REPO_DIR="/home/harshilb/3d_flowmatch_actor"
cd "$REPO_DIR"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

TASKS=(
    place_cups close_jar insert_onto_square_peg
    light_bulb_in meat_off_grill open_drawer
    place_shape_in_shape_sorter place_wine_at_rack_location
    push_buttons put_groceries_in_cupboard
    put_item_in_drawer put_money_in_safe reach_and_drag
    slide_block_to_color_target stack_blocks stack_cups
    sweep_to_dustpan_of_size turn_tap
)

TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"

# CHECKPOINT and RUN_LOG_DIR must be passed via --export
: "${CHECKPOINT:?'CHECKPOINT not set — pass via --export=CHECKPOINT=...'}"
: "${RUN_LOG_DIR:?'RUN_LOG_DIR not set — pass via --export=RUN_LOG_DIR=...'}"

OUTPUT_DIR="${REPO_DIR}/eval_logs/Orbital/${RUN_LOG_DIR}"
OUTPUT_FILE="${OUTPUT_DIR}/results_${TASK}.json"

mkdir -p "$OUTPUT_DIR" logs

echo "Task: $TASK"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_FILE"

if [[ -f "$OUTPUT_FILE" ]]; then
    echo "[SKIP] Results already exist."
    exit 0
fi

# ── Shared model config ───────────────────────────────────────────────────────
eval_data_dir=/grogu/user/harshilb/orbital_train.zarr
val_instructions=instructions/peract/instructions.json
cameras_file=instructions/orbital_cameras_grouped.json
task_group_mapping_file=instructions/task_group_mapping.json

dataset=OrbitalWrist
image_size="256,256"
fov_deg=60.0
miscalibration_noise_level=null
max_steps=25
prediction_len=1
num_history=3
max_tries=1

model_type=denoise3d
bimanual=false
backbone=clip
finetune_backbone=false
finetune_text_encoder=false
fps_subsampling_factor=4
C=120
num_attn_heads=8
num_vis_instr_attn_layers=2
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

headless=true
collision_checking=false
seed=0

xvfb-run -a python online_evaluation_rlbench/evaluate_policy.py \
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
    checkpoint=$CHECKPOINT \
    output_file=$OUTPUT_FILE \
    task=$TASK \
    headless=$headless \
    collision_checking=$collision_checking \
    seed=$seed

echo "Done at: $(date)"
