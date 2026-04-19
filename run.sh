# xvfb-run -a python scripts/rlbench/visualize_cameras_rerun.py \
#     --task close_jar \
#     --out  camera_viz.rrd


xvfb-run -a python scripts/rlbench/visualize_cameras_rerun.py \
    --task close_jar --cameras-file instructions/orbital_cameras.json --out camera_viz.rrd           


#   xvfb-run -a python scripts/rlbench/visualize_cameras_rerun.py \
#       --task bimanual_lift_tray --bimanual \
#       --out  camera_viz.rrdwha