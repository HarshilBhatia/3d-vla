# xvfb-run -a python scripts/rlbench/visualize_cameras_rerun.py \
#     --task close_jar \
#     --out  camera_viz.rrd


# xvfb-run -a python scripts/rlbench/visualize_cameras_rerun.py \
#     --task close_jar --cameras-file instructions/orbital_cameras.json --out camera_viz.rrd           


#   xvfb-run -a python scripts/rlbench/visualize_cameras_rerun.py \
#       --task bimanual_lift_tray --bimanual \
#       --out  camera_viz.rrdwha

export PYTHONPATH=/root/3d_flowmatch_actor:$PYTHONPATH                                                                    
export COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04                                                          
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH                                                                 
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT 
                                                                                       

 xvfb-run -a bash scripts/eval/online_eval.sh \
      --checkpoint best.pth \
      --tasks "stack_blocks" \
      --extra "data_dir=/ocean/projects/cis240058p/hbhatia1/3d-vla/data/peract_G1_data camera_groups=G1"



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


unset QT_QPA_PLATFORM
mkdir -p /run/user/27491 && chmod 700 /run/user/27491
export XDG_RUNTIME_DIR=/run/user/27491

#  xvfb-run -a --server-args="-screen 0 1280x1024x24 +extension GLX +render"  bash scripts/eval/online_eval.sh \
#       --checkpoint  \
#       --tasks "reach_and_drag" \
#       --extra "data_dir=/root/peract_G1_data camera_groups=G1"


xvfb-run -a --server-args="-screen 0 1280x1024x24 +extension GLX +render" bash scripts/eval/online_eval_bimanual.sh \
      --checkpoint /grogu/user/harshilb/train_logs/exp/final_default_full/best.pth \
      --run-log-dir out_test/
#
