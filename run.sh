# apptainer exec --nv /ocean/projects/cis240058p/hbhatia1/containers/3dfa-sandbox.sif \
# xvfb-run -a python scripts/rlbench/visualize_cameras_rerun.py \
#     --task close_jar --cameras-file orbital_cameras.json \
#     --ood-cameras-file ood_camera.json --out camera_viz.rrd

#   xvfb-run -a python scripts/rlbench/visualize_cameras_rerun.py \
#       --task bimanual_lift_tray --bimanual \
#       --out  camera_viz.rrd


#   rsync -avz --progress \
#       /ocean/projects/cis240058p/hbhatia1/3d-vla/data/orbital_train_v2.zarr \stric
#       harshilb@grogu.ri.cmu.edu:/grogu/user/harshilb/

# REPO_DIR="/ocean/projects/cis240058p/hbhatia1/3d-vla"
# CONTAINER="/ocean/projects/cis240058p/hbhatia1/containers/3dfa-sandbox.sif"
# COPPELIASIM_ROOT="${REPO_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"

# xvfb-run -a \
#     apptainer exec \
#         --env "COPPELIASIM_ROOT=${COPPELIASIM_ROOT}" \
#         --env "LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}" \
#         --env "QT_QPA_PLATFORM_PLUGIN_PATH=${COPPELIASIM_ROOT}" \
#         --env "PYTHONPATH=${REPO_DIR}/RLBench:${REPO_DIR}" \
#         --bind "${REPO_DIR}:${REPO_DIR}" \
#         "${CONTAINER}" \
#         python3 "${REPO_DIR}/RLBench/tools/dataset_generator.py" \
#             --save_path     "${REPO_DIR}/data/peract_raw_ood" \
#             --tasks         close_jar open_drawer turn_tap \
#             --episodes_per_task 100 \
#             --all_variations False \
#             --variations 1 \
#             --camera_file   "${REPO_DIR}/ood_camera.json"

#   find /ocean/projects/cis240058p/hbhatia1/3d-vla/data/peract_raw_ood \                                                                                                                                                     
#       -mindepth 1 ! -name "low_dim_obs.pkl" ! -type d -delete && \
#   find /ocean/projects/cis240058p/hbhatia1/3d-vla/data/peract_raw_ood \                                                                                                                                                     
#       -mindepth 1 -type d -empty -delete                                                                                                                                                                                    
                                                      

  rsync -avz --progress \
      /ocean/projects/cis240058p/hbhatia1/3d-vla/data/peract_raw_ood/a.zip \
      harshilb@euclid.ri.cs.cmu.edu:/home/harshilb/work/3d-vla/