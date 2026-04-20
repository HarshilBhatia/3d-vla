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