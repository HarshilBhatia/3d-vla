#!/bin/bash
# copy all folders from /grogu/user/harshilb/train_logs/Peract2 to /home/harshilb/3d_flowmatch_actor/train_logs

folders=(
    # "2scene-ComRoPE-front_cam-false-traj_scene_rope-true"
    # "2scene-ComRoPE-front_cam-true-traj_scene_rope-true"
    # "2scene-cam-token-traj_scene_ropetrue-front-cam-true"
    # "2scene-LEFalse-traj_scene_ropetrue-front-cam-false"
    # "2scene-LEFalse-traj_scene_ropefalse-front-cam-true"
    "full-3dfa-rope_type-normal-pred-false-front-false"
    "baseline-rope_type-normal-pred-false-front-true"
)

for folder in "${folders[@]}"; do
    cp -r /grogu/user/harshilb/train_logs/Peract2/"$folder" /home/harshilb/3d_flowmatch_actor/train_logs/"$folder"
done