#!/usr/bin/env bash
# Run one eval episode and save colored point clouds for inspection.
#
# Usage (from repo root):
#   xvfb-run -a bash scripts/eval/debug_pcd.sh \
#       --checkpoint train_logs/exp/my_run/last.pth \
#       --task bimanual_lift_tray
#
# Optional:
#   --debug-pcd-dir  /tmp/debug_pcd          (default: /tmp/debug_pcd)
#   --extra          "fps_subsampling_factor=4"
#
# Output:
#   <debug-pcd-dir>/pcd_NNNN.ply      full point cloud with RGB
#   <debug-pcd-dir>/fps_pcd_NNNN.ply  density-subsampled cloud with RGB
#   <debug-pcd-dir>/pcd_NNNN.npz      same data as numpy arrays
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CHECKPOINT=""
TASK="bimanual_lift_tray"
DEBUG_PCD_DIR="/tmp/debug_pcd"
EXTRA_OVERRIDES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)    CHECKPOINT="$2";    shift 2 ;;
        --task)          TASK="$2";          shift 2 ;;
        --debug-pcd-dir) DEBUG_PCD_DIR="$2"; shift 2 ;;
        --extra)         EXTRA_OVERRIDES="$2"; shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT" ]]; then
    echo "[ERROR] --checkpoint is required."
    echo "Usage: xvfb-run -a bash scripts/eval/debug_pcd.sh --checkpoint <path> --task <task>"
    exit 1
fi

mkdir -p "$DEBUG_PCD_DIR"
echo "[INFO] checkpoint    : $CHECKPOINT"
echo "[INFO] task          : $TASK"
echo "[INFO] debug_pcd_dir : $DEBUG_PCD_DIR"
echo

python "${REPO_ROOT}/online_evaluation_rlbench/evaluate_policy.py" \
    data=full \
    experiment=default \
    data_dir=/grogu/user/harshilb/datasets/peract2_raw/peract2_test \
    checkpoint="$CHECKPOINT" \
    task="$TASK" \
    max_tries=1 \
    headless=true \
    output_file="${DEBUG_PCD_DIR}/results.json" \
    debug_pcd_dir="$DEBUG_PCD_DIR" \
    $EXTRA_OVERRIDES

echo
echo "PLY files saved to $DEBUG_PCD_DIR — open with MeshLab or:"
echo "  python -c \"import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('${DEBUG_PCD_DIR}/pcd_0000.ply')])\""
