exp=peract2
tasks=(
    bimanual_pick_plate
)


# Testing arguments

checkpoint_alias=my_awesome_peract2_model  # or something ugly
# like: denoise3d-Peract2_3dfront_3dwrist-C120-B64-lr1e-4-constant-H3-rectified_flow

max_tries=2
max_steps=25
headless=true
collision_checking=false
seed=0

# Dataset arguments
data_dir=peract2_test/
dataset=Peract2_3dfront_3dwrist
image_size=256,256

# Model arguments
model_type=denoise3d
bimanual=true
prediction_len=1

backbone=clip
fps_subsampling_factor=4

embedding_dim=120
num_attn_heads=8
num_vis_instr_attn_layers=3
num_history=3

num_shared_attn_layers=4
relative_action=false
rotation_format=quat_xyzw
denoise_timesteps=5
denoise_model=rectified_flow



checkpoint=/home/harshilb/work/3d-vla/grogu_train_logs/baseline-rope_type-normal-pred-false-front-true/best.pth

# /home/harshilb/work/3d-vla/grogu_train_logs/2scene-LEFalse-traj_scene_ropefalse-front-cam-true/best.pth

# /home/harshilb/work/3d-vla/grogu_train_logs/full-3dfa-rope_type-normal-pred-false-front-false/best.pth





# /home/harshilb/work/3d-vla/grogu_train_logs/baseline-rope_type-normal-pred-false-front-true/best.pth

# /home/harshilb/work/3d-vla/grogu_train_logs/2scene-LEFalse-traj_scene_ropefalse-front-cam-true/best.pth

# /home/harshilb/work/3d-vla/grogu_train_logs/2scene-ComRoPE-front_cam-false-traj_scene_rope-true/best.pth

# /home/harshilb/work/3d-vla/grogu_train_logs/Peract2/2scene_deltaM_new-front_cam-true-cam_token-true-traj_scene_rope-true/best.pth
# /home/harshilb/work/3d-vla/grogu_train_logs/Peract2/2scene_deltaM_new-front_cam-true-cam_token-true-traj_scene_rope-true/best.pth # old best                      

checkpoint_dir=$(dirname "$checkpoint")

learn_extrinsics=false
traj_scene_rope=true

front_camera_frame=false
predict_extrinsics=false

extrinsics_prediction_mode=delta_m

rope_type=normal


use_com_rope=false
com_rope_block_size=3
com_rope_num_axes=3
com_rope_init_std=0.02


CUDA_VISIBLE_DEVICES=1 xvfb-run -a python online_evaluation_rlbench/evaluate_policy.py \
    --checkpoint $checkpoint \
    --task ${tasks[$i]} \
    --max_tries $max_tries \
    --max_steps $max_steps \
    --headless $headless \
    --collision_checking $collision_checking \
    --seed $seed \
    --data_dir $data_dir \
    --dataset $dataset \
    --image_size $image_size \
    --output_file $checkpoint_dir/seed$seed/${tasks[$i]}/eval.json \
    --model_type $model_type \
    --bimanual $bimanual \
    --prediction_len $prediction_len \
    --backbone $backbone \
    --fps_subsampling_factor $fps_subsampling_factor \
    --embedding_dim $embedding_dim \
    --num_attn_heads $num_attn_heads \
    --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
    --num_history $num_history \
    --num_shared_attn_layers $num_shared_attn_layers \
    --relative_action $relative_action \
    --rotation_format $rotation_format \
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model \
    --learn_extrinsics $learn_extrinsics \
    --traj_scene_rope $traj_scene_rope \
    --front_camera_frame $front_camera_frame \
    --predict_extrinsics $predict_extrinsics \
    --extrinsics_prediction_mode $extrinsics_prediction_mode \
    --rope_type $rope_type \
    --use_com_rope $use_com_rope \
    --com_rope_block_size $com_rope_block_size \
    --com_rope_num_axes $com_rope_num_axes \
    --com_rope_init_std $com_rope_init_std \

python online_evaluation_rlbench/collect_results.py \
    --folder $checkpoint_dir/seed$seed/