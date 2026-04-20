#!/usr/bin/env bash
# Evaluate grogu_train_logs/best.pth (deltaM + large miscalibration) on all tasks.
# Usage: bash scripts/rlbench/eval_orbital_grogu_best.sh

main_dir=Orbital
run_log_dir="grogu_best_miscal_large_deltaM"

data_dir=/home/harshilb/work/3d-vla/ood
val_instructions=instructions/peract/instructions.json

dataset=OrbitalWrist
cameras_file=orbital_cameras_grouped.json
task_group_mapping_file=task_group_mapping.json
fov_deg=60.0
miscalibration_noise_level=large

image_size="256,256"
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

checkpoint=/home/harshilb/work/3d-vla/grogu_train_logs/best.pth

headless=true
collision_checking=false
seed=0

# tasks=(
#     place_cups
#     close_jar
#     insert_onto_square_peg
#     light_bulb_in
#     meat_off_grill
#     open_drawer
#     place_shape_in_shape_sorter
#     place_wine_at_rack_location
#     push_buttons
#     put_groceries_in_cupboard
#     put_item_in_drawer
#     put_money_in_safe
#     reach_and_drag
#     slide_block_to_color_target
#     stack_blocks
#     stack_cups
#     sweep_to_dustpan_of_size
#     turn_tap
# )

tasks=(
    open_drawer
    turn_tap
)

mkdir -p eval_logs/${main_dir}/${run_log_dir}
mkdir -p eval_logs/${main_dir}/${run_log_dir}/logs

run_task() {
    local task=$1
    local gpu=$2
    local output_file=eval_logs/${main_dir}/${run_log_dir}/${task}.json
    local log_file=eval_logs/${main_dir}/${run_log_dir}/logs/${task}.log

    echo "[GPU $gpu] Starting $task ..."
    CUDA_VISIBLE_DEVICES=$gpu xvfb-run -a python online_evaluation_rlbench/evaluate_policy.py \
        val_instructions=$val_instructions \
        data_dir=$data_dir \
        dataset=$dataset \
        cameras_file=$cameras_file \
        task_group_mapping_file=$task_group_mapping_file \
        fov_deg=$fov_deg \
        miscalibration_noise_level=$miscalibration_noise_level \
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
        seed=$seed 
    echo "[GPU $gpu] Done $task — exit $?"
}

# Run tasks in batches of 2 (one per GPU)
num_gpus=2
pids=()
gpu_slots=(0 1)

for i in "${!tasks[@]}"; do
    task=${tasks[$i]}
    gpu=${gpu_slots[$((i % num_gpus))]}

    # If both GPU slots are busy, wait for the one we're about to reuse
    if [ $i -ge $num_gpus ]; then
        wait ${pids[$((i - num_gpus))]}
    fi

    run_task "$task" "$gpu" &
    pids[$i]=$!
done

# Wait for remaining jobs
for pid in "${pids[@]}"; do
    wait $pid
done

echo "All tasks done. Results in eval_logs/${main_dir}/${run_log_dir}/"
