# xvfb-run -a python scripts/rlbench/visualize_cameras_rerun.py \
#     --task close_jar \
#     --out  camera_viz.rrd


# xvfb-run -a python scripts/rlbench/visualize_cameras_rerun.py \
#     --task close_jar --cameras-file instructions/orbital_cameras.json --out camera_viz.rrd           


#   xvfb-run -a python scripts/rlbench/visualize_cameras_rerun.py \
#       --task bimanual_lift_tray --bimanual \
#       --out  camera_viz.rrdwha


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