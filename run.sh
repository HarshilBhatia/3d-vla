
export PYTHONPATH=/root/3d_flowmatch_actor:$PYTHONPATH                                                                    
export COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04                                                          
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH                                                                 
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT 


unset QT_QPA_PLATFORM
mkdir -p /run/user/27491 && chmod 700 /run/user/27491
export XDG_RUNTIME_DIR=/run/user/27491                                                                  

# bash scripts/eval/online_eval_peract_collected.sh 

# xvfb-run -a python scripts/orbital_cameras/collect.py \
#     --task light_bulb_in \
#     --groups G6  \
#     --n-episodes 1 \
#     --save-path data/orbital_rollouts_test \
#     --cameras-file instructions/orbital_cameras_grouped.json

# CHECKPOINT='/root/3d_flowmatch_actor/train_logs/Orbital/open_drawer_default_G1/step_45000.pth'
CHECKPOINT='/root/3d_flowmatch_actor/train_logs/Orbital/open_drawer_default_G1_miscal/step_45000.pth'
# CHECKPOINT='/root/3d_flowmatch_actor/train_logs/Orbital/3dfa_run/step_160000.pth'

# CHECKPOINT='/root/3d_flowmatch_actor/train_logs/Orbital/open_drawer_G1/interm40000.pth'
# CHECKPOINT='/root/3d_flowmatch_actor/train_logs/Peract/peract_collected/interm40000.pth'
# /root/3d_flowmatch_actor/train_logs/Orbital/open_drawer_G1/interm40000.pth'

xvfb-run -a --server-args="-screen 0 1280x1024x24 +extension GLX +render" bash scripts/eval/eval_orbital.sh \
    checkpoint=$CHECKPOINT\
    task="open_drawer" \
    "data_dir=/grogu/user/harshilb/orbital_rollouts_mini/"\
    "camera_groups=G1" 
    
    # num_history=3 \
    # use_recursive_set_encoder=true \
    # recursive_set_encoder_ncam=3 \
    # embedding_dim=192           


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

# python data/processing/convert_to_zarr/orbital_to_zarr.py \
#     --root /grogu/user/harshilb/orbital_rollouts_mini/ \
#     --out /grogu/user/harshilb/1task_new --overwrite 


# python data/processing/convert_to_zarr/peract_to_zarr.py \
    # --root /grogu/user/harshilb/peract_rollouts_mini/ \
    # --tgt /grogu/user/harshilb/peract_mini.zarr --overwrite

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


