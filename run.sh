# xvfb-run -a python scripts/rlbench/visualize_cameras_rerun.py \
#     --task close_jar \
#     --out  camera_viz.rrd


# xvfb-run -a python scripts/rlbench/visualize_cameras_rerun.py \
#     --task close_jar --cameras-file instructions/orbital_cameras.json --out camera_viz.rrd           


#   xvfb-run -a python scripts/rlbench/visualize_cameras_rerun.py \
#       --task bimanual_lift_tray --bimanual \
#       --out  camera_viz.rrdwha


 xvfb-run -a bash scripts/eval/online_eval.sh \
      --checkpoint /home/harshilb/work/3d-vla/grogu_train_logs/best.pth \
      --tasks "meat_off_grill" \
      --extra "data_dir=/home/harshilb/work/3d-vla/peract_G1_data camera_groups=G1"