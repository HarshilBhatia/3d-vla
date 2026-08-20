
export PYTHONPATH=/root/3d_flowmatch_actor:$PYTHONPATH                                                                    
export COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04                                                          
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH                                                                 
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT 


unset QT_QPA_PLATFORM
mkdir -p /run/user/27491 && chmod 700 /run/user/27491
export XDG_RUNTIME_DIR=/run/user/27491                                                                  


# POLICY_SOCKET_PATH=/tmp/policy.sock POLICY_SERVER_TYPE=openpi \
# xvfb-run -a --server-args="-screen 0 1280x1024x24 +extension GLX +render" \
#   python online_evaluation_rlbench/evaluate_policy_external.py \
#       dataset=OrbitalWrist bimanual=false task=open_drawer \
#       output_file=eval_logs/external/open_drawer.json \
#       data_dir=/grogu/user/harshilb/low_dim_demos/ \
#       cameras_file=instructions/orbital_cameras_grouped.json \
#       task_group_mapping_file=instructions/task_group_mapping_subset.json \
#       spawn_camera_group=G5

# POLICY_SOCKET_PATH=/tmp/policy.sock POLICY_SERVER_TYPE=openpi \
# xvfb-run -a --server-args="-screen 0 1280x1024x24 +extension GLX +render" \
#       python online_evaluation_rlbench/evaluate_policy_external.py \
#       dataset=OrbitalWrist bimanual=false task=open_drawer \
#       output_file=eval_logs/external/open_drawer.json \
#       data_dir=/grogu/user/harshilb/low_dim_demos/ \
#       cameras_file=instructions/orbital_cameras_grouped.json \
#       task_group_mapping_file=instructions/task_group_mapping_subset.json \
#       camera_groups="GT" \
    #   spawn_camera_group=G5

# xvfb-run -a python scripts/orbital_cameras/collect.py \
#     --task open_drawer --groups G7 \
#     --cameras-file instructions/orbital_cameras_grouped.json \
#     --video-only --video-dir debug_videos/

# xvfb-run -a python scripts/orbital_cameras/collect.py \
#     --task light_bulb_in \
#     --groups G6  \
#     --n-episodes 1 \
#     --save-path data/orbital_rollouts_test \
#     --cameras-file instructions/orbital_cameras_grouped.json

# CHECKPOINT='/root/3d_flowmatch_actor/train_logs/Orbital/open_drawer_default_G1/step_45000.pth'
# CHECKPOINT='/root/3d_flowmatch_actor/train_logs/Orbital/open_drawer_default_G1_miscal/step_45000.pth'
# CHECKPOINT='/root/3d_flowmatch_actor/train_logs/Orbital/3dfa_run/step_160000.pth'

# CHECKPOINT='/root/3d_flowmatch_actor/train_logs/Orbital/open_drawer_G1/interm40000.pth'
# CHECKPOINT='/root/3d_flowmatch_actor/train_logs/Peract/peract_collected/interm40000.pth'
# /root/3d_flowmatch_actor/train_logs/Orbital/open_drawer_G1/interm40000.pth'

# xvfb-run -a --server-args="-screen 0 1280x1024x24 +extension GLX +render" bash scripts/eval/eval_orbital.sh \
#     checkpoint=$CHECKPOINT\
#     task="open_drawer" \
#     "data_dir=/grogu/user/harshilb/orbital_rollouts_mini/"\
#     "camera_groups=G1" 
    
    # num_history=3 \
    # use_recursive_set_encoder=true \
    # recursive_set_encoder_ncam=3 \
    # embedding_dim=192           


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

# python scripts/helpers/vis_low_dim_obs.py /grogu/user/harshilb/orbital_rollouts_mini/open_drawer/G1/episode_0/

# python data/processing/convert_to_zarr/orbital_to_zarr.py \
#     --root /grogu/user/harshilb/multi_cam_val \
#     --out /grogu/user/harshilb/multi_cam/val.zarr --overwrite


# python data/processing/convert_to_zarr/orbital_to_zarr.py \
#     --root /grogu/user/harshilb/full_rollouts_merged \
#     --out /grogu/datasets/hbhatia/peract_orb_new/ \
#     --train-episodes 33 \
#     --overwrite



# python orbital_to_zarr.py \
#     --root /path/to/data \
#     --out /path/to/output_dir \
#     --overwrite

# xvfb-run -a python scripts/orbital_cameras/collect_low_dim_only.py \
#     --tasks inset_onto_squ \
#     --n-episodes 30 \
#     --save-path data/orbital_low_dim
    
python data/processing/convert_to_zarr/peract_collected_to_zarr.py \
    --root /grogu/datasets/hbhatia/peract_rollouts_new/ \
    --tgt /grogu/user/harshilb/peract_subset/train.zarr --overwrite

# python data/processing/convert_to_zarr/peract_to_zarr.py \
#     --root /grogu/datasets/hbhatia/peract_rollouts_new/ \
#     --out /grogu/user/harshilb/peract_subset/ \
#     --overwrite

# python data/processing/convert_to_zarr/peract_collected_to_zarr.py \
#     --root /grogu/user/harshilb/peract_rollouts/ \
#     --tgt /grogu/user/harshilb/peract_mini.zarr \
#     --overwrite


# apptainer shell --nv --fakeroot --writable --bind /usr/bin/xvfb-run:/usr/bin/xvfb-run  --bind /grogu/datasets/hbhatia/:/grogu/datasets/hbhatia/  --bind /grogu/user/harshilb/:/grogu/user/harshilb/ --env PATH=/root/miniconda3/envs/3dfa/bin:$PATH  my_eval_env

# apptainer shell --nv --fakeroot --bind /scratch:/scratch --bind /grogu/user/harshilb/:/grogu/user/harshilb/ 3dfa_flash


# rsync -avzP /grogu/user/harshilb/open_drawer.zip  hbhatia1@login.delta.ncsa.illinois.edu:/work/hdd/bgkz/hbhatia1

# apptainer shell --nv --fakeroot --writable --network host --bind /usr/bin/xvfb-run:/usr/bin/xvfb-run  --bind /grogu/datasets/hbhatia/:/grogu/datasets/hbhatia/  --bind /grogu/user/harshilb/:/grogu/user/harshilb/ --env PATH=/root/miniconda3/envs/3dfa/bin:$PATH  my_eval_env

#olp_8dvL4vVgioXAnRBdxB0tdS7e336VCj1I22bb