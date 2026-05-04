#!/usr/bin/env bash
# Run a single training step and save colored point clouds for inspection.
#
# Usage (from repo root):
#   bash scripts/train/debug_pcd.sh \
#       --checkpoint train_logs/exp/my_run/last.pth
#
# Optional:
#   --debug-pcd-dir  /tmp/debug_pcd   (default: /tmp/debug_pcd)
#   --experiment     camtoken_deltaM  (default: default)
#   --extra          "fps_subsampling_factor=4"
#
# Output:
#   <debug-pcd-dir>/pcd_0000.ply      full point cloud with RGB
#   <debug-pcd-dir>/fps_pcd_0000.ply  density-subsampled cloud with RGB
#   <debug-pcd-dir>/pcd_0000.npz      same data as numpy arrays
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DEBUG_PCD_DIR='eval_logs/'
data=orbital
main_dir=Orbital
experiment=baseline

run_log_dir=rndm
checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth
ngpus=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
MASTER_PORT=$((27500 + RANDOM % 1000))



WANDB_API_KEY=$WANDB_API_KEY torchrun --nproc_per_node $ngpus --master_port $MASTER_PORT \
    main.py \
    data=$data \
    experiment=$experiment \
    exp_log_dir=$main_dir \
    run_log_dir=$run_log_dir \
    checkpoint=$checkpoint \
    train_iters=1 \
    val_freq=999999 \
    train_data_dir=/grogu/user/harshilb/1task_new.zarr \
    eval_data_dir=/grogu/user/harshilb/1task_new.zarr \
    +debug_pcd_dir="$DEBUG_PCD_DIR" \

echo
echo "PLY files saved to $DEBUG_PCD_DIR — open with MeshLab or:"
echo "  python -c \"import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('${DEBUG_PCD_DIR}/pcd_0000.ply')])\""
