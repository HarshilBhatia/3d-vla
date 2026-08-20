#!/bin/bash
# LIBERO-plus Camera-Viewpoint sweep for traj_hist1 (376 variants, 1 ep each).
# Extrinsics are read live from the perturbed sim → "correct extrinsics" mode.
# Chunked per 10 tasks with hard timeout; resumable via per-chunk json.
set -u
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH=/k8s-nfs/harsvbha/3dfa/libero/.libero-plus
export PYTHONPATH=/k8s-nfs/harsvbha/3dfa/libero/LIBERO-plus
PY=/k8s-nfs/harsvbha/3dfa/venv/bin/python
EVAL=/k8s-nfs/harsvbha/3dfa/eval/plus_camera
CK=/k8s-nfs/harsvbha/3dfa/repo/train_logs/Libero/libero_spatial_traj/last.pth
mkdir -p $EVAL
cd /k8s-nfs/harsvbha/3dfa/repo

run_range() { # gpu start end
  local gpu=$1 start=$2 end=$3
  local t=$start
  while [ $t -le $end ]; do
    local hi=$((t + 9)); [ $hi -gt $end ] && hi=$end
    local ids="${t}-${hi}"
    local out=$EVAL/cam_${t}_${hi}.json
    if [ ! -s "$out" ]; then
      timeout -k 30 3600 env CUDA_VISIBLE_DEVICES=$gpu \
        $PY online_evaluation_libero/evaluate_policy.py \
          checkpoint=$CK prediction_len=50 task=$ids num_demos=1 \
          max_steps=12 output_file=$out \
          >> /tmp/plus_cam_gpu${gpu}.log 2>&1
      echo "[sweep] chunk $t-$hi exit=$?" >> /tmp/plus_cam_gpu${gpu}.log
    fi
    t=$((hi + 1))
  done
}

run_range 0 608 795 &
run_range 1 796 983 &
wait
echo "[sweep] camera axis done"
