
export PYTHONPATH=/root/3d_flowmatch_actor:$PYTHONPATH                                                                    
export COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04                                                          
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH                                                                 
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT 


unset QT_QPA_PLATFORM
mkdir -p /run/user/27491 && chmod 700 /run/user/27491
export XDG_RUNTIME_DIR=/run/user/27491                                                                  

# python scripts/analyze_pkl_size.py /grogu/user/harshilb/orbital_rollouts/open_drawer/G1/episode_0/low_dim_obs.pkl                                                          

# xvfb-run -a python scripts/orbital_cameras/collect.py \
#     --task light_bulb_in \
#     --groups G6  \
#     --n-episodes 1 \
#     --save-path data/orbital_rollouts_test \
#     --cameras-file instructions/orbital_cameras_grouped.json

#  xvfb-run -a --server-args="-screen 0 1280x1024x24 +extension GLX +render" bash scripts/eval/eval_orbital.sh \
#       checkpoint=/root/3d_flowmatch_actor/train_logs/Orbital/open_drawer_test/last.pth \
#       task="light_bulb_in" \
#       "data_dir=/grogu/user/harshilb/orbital_rollouts"\
#       "camera_groups=G1" \
#       "miscalibration_noise_level=medium"
    #   --extra "data_dir=/ocean/projects/cis240058p/hbhatia1/3d-vla/data/peract_G1_data camera_groups=G1"


# python scripts/print_rollout.py data/orbital_rollouts/insert_onto_square_peg/G3/episode_0 

# python scripts/print_rollout.py data/orbital_rollouts/insert_onto_square_peg/G3/episode_0 --all-frames --out traj.png                      

# python -c "                                                                                                                                
# import pickle                                                                                                                              
# with open('data/orbital_rollouts/insert_onto_square_peg/G3/episode_0/low_dim_obs.pkl', 'rb') as f:
#     obs = pickle.load(f)                                                                                                                   
# print(type(obs))                                                                                                                           
# print('variation_number:', getattr(obs, 'variation_number', 'NOT SET'))
# print(dir(obs))                                                                                                                            
# "

python data/processing/convert_to_zarr/orbital_to_zarr.py \
    --root /grogu/user/harshilb/orbital_rollouts/ \
    --out /grogu/user/harshilb/orbital_rollouts/4task_new.zarr \
    --overwrite


# python data/processing/convert_to_zarr/peract_to_zarr.py \
#     --root /grogu/user/harshilb/peract_rollouts/ \
#     --tgt /grogu/user/harshilb/peract_mini.zarr --overwrite

# python data/processing/convert_to_zarr/peract_collected_to_zarr.py \
#     --root /grogu/user/harshilb/peract_rollouts/ \
#     --tgt /grogu/user/harshilb/peract_mini.zarr \
#     --overwrite
    

# apptainer exec \
#       --nv \
#       --env "COPPELIASIM_ROOT=${COPPELIASIM_ROOT}" \
#       --env "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}" \
#       --env "QT_QPA_PLATFORM_PLUGIN_PATH=${COPPELIASIM_ROOT}" \
#       --bind /ocean/projects/cis240058p/hbhatia1/3d-vla:/ocean/projects/cis240058p/hbhatia1/3d-vla \
#       /ocean/projects/cis240058p/hbhatia1/containers/3dfa-sandbox.sif \
#       bash


#   xvfb-run -a bash scripts/eval/online_eval.sh \                                                                                                                                                                            
#       --checkpoint /path/to/your/checkpoint.pth \
#       --tasks "stack_blocks" \                                                                                                                                                                                              
#       --extra "eval_data_dir=/ocean/projects/cis240058p/hbhatia1/3d-vla/data/peract_G1_data camera_groups=G1"



#  xvfb-run -a --server-args="-screen 0 1280x1024x24 +extension GLX +render"  bash scripts/eval/online_eval.sh \
#       --checkpoint  \
#       --tasks "reach_and_drag" \

# data/orbital_rollouts_test/light_bulb_in/G6/

# xvfb-run -a --server-args="-screen 0 1280x1024x24 +extension GLX +render" bash scripts/eval/online_eval_bimanual.sh \
#       --checkpoint /grogu/user/harshilb/train_logs/exp/final_default_full/best.pth \
#       --run-log-dir out_test/
# #

# set -euo pipefail

# REPO_ROOT="."

# TASK="open_drawer"
# CHECKPOINT="${REPO_ROOT}/train_logs/Peract/peract_collected/last.pth"
# OUTPUT_FILE="${REPO_ROOT}/eval_logs/Peract/peract_collected/results_${TASK}.json"

# mkdir -p "$(dirname "$OUTPUT_FILE")" logs

# xvfb-run -a python "${REPO_ROOT}/online_evaluation_rlbench/evaluate_policy.py" \
#     dataset=PeractCollected \
#     data_dir=/grogu/user/harshilb/peract_rollouts/ \
#     val_instructions=instructions/peract/instructions.json \
#     "image_size='128,128'"  \
#     bimanual=false \
#     model_type=denoise3d \
#     backbone=clip \
#     finetune_backbone=false \
#     finetune_text_encoder=false \
#     fps_subsampling_factor=5 \
#     embedding_dim=120 \
#     num_attn_heads=8 \
#     num_vis_instr_attn_layers=3 \
#     num_shared_attn_layers=4 \
#     relative_action=false \
#     rotation_format=quat_xyzw \
#     denoise_timesteps=5 \
#     denoise_model=rectified_flow \
#     learn_extrinsics=false \
#     predict_extrinsics=false \
#     traj_scene_rope=true \
#     rope_type=normal \
#     sa_blocks_use_rope=true \
#     max_steps=25 \
#     prediction_len=1 \
#     num_history=3 \
#     max_tries=3 \
#     headless=true \
#     collision_checking=false \
#     seed=0 \
#     checkpoint=$CHECKPOINT \
#     output_file=$OUTPUT_FILE \
#     task=$TASK
