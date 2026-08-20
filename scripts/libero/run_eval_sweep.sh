#!/bin/bash
# LIBERO-Spatial eval sweep on d064: 3 checkpoints x 10 tasks x 10 episodes.
# One subprocess per task with a hard `timeout` kill — a C-level MuJoCo/EGL
# hang survives in-process SIGALRM, so isolation must be at the process level.
# Resumable: tasks with an existing non-empty json are skipped.
set -u
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH=/k8s-nfs/harsvbha/3dfa/libero/.libero-shared
export PYTHONPATH=/k8s-nfs/harsvbha/3dfa/libero/LIBERO
PY=/k8s-nfs/harsvbha/3dfa/venv/bin/python
EVAL=/k8s-nfs/harsvbha/3dfa/eval
cd /k8s-nfs/harsvbha/3dfa/repo

run_model() { # name ckpt prediction_len gpu
  local name=$1 ck=$2 plen=$3 gpu=$4
  for t in 0 1 2 3 4 5 6 7 8 9; do
    local out=$EVAL/${name}_t${t}.json
    [ -s "$out" ] && continue
    timeout -k 30 1800 env CUDA_VISIBLE_DEVICES=$gpu \
      $PY online_evaluation_libero/evaluate_policy.py \
        checkpoint=$ck prediction_len=$plen task=$t num_demos=10 \
        max_steps=12 output_file=$out \
        >> $EVAL/${name}.log 2>&1
    echo "[sweep] $name task $t exit=$?" >> $EVAL/${name}.log
  done
}

run_model traj_hist1 /k8s-nfs/harsvbha/3dfa/repo/train_logs/Libero/libero_spatial_traj/last.pth 50 0 &
run_model traj_hist3 /k8s-nfs/harsvbha/3dfa/repo/train_logs/Libero/libero_spatial_traj_hist3/last.pth 50 1 &
wait
run_model keypose_hist3 /k8s-nfs/harsvbha/3dfa/repo/train_logs/Libero/libero_spatial_base_hist3/last.pth 1 0
echo "[sweep] all done"
