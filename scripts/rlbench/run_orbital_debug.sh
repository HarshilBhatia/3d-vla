#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_orbital_debug.sh
#
# Generate 54 debug videos + zarrs (18 tasks × 3 groups each).
# One CoppeliaSim launch per task — all 3 groups collected in that session.
# 18 launches instead of 54, saving ~66% of startup overhead.
#
# Output per (task, group) pair:
#   debug_videos/${TASK}_${GROUP}.mp4        ← 4-panel video
#   debug_videos/${TASK}_${GROUP}.mp4.zarr/  ← single-episode zarr
#
# Usage (from repo root):
#   xvfb-run -a bash scripts/rlbench/run_orbital_debug.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAPPING="${REPO_ROOT}/task_group_mapping.json"
CAMERAS_FILE="${REPO_ROOT}/orbital_cameras_grouped.json"
VIDEO_DIR="${REPO_ROOT}/debug_videos"
SCRIPT="${REPO_ROOT}/scripts/rlbench/collect_orbital_rollouts.py"

mkdir -p "${VIDEO_DIR}"

if [ ! -f "${MAPPING}" ]; then
    echo "[ERROR] task_group_mapping.json not found."
    echo "        Run: python scripts/rlbench/create_task_group_mapping.py"
    exit 1
fi

python3 - <<PYEOF
import json, os, subprocess, sys

repo_root    = "${REPO_ROOT}"
mapping_path = "${MAPPING}"
cameras_file = "${CAMERAS_FILE}"
video_dir    = "${VIDEO_DIR}"
script       = "${SCRIPT}"

with open(mapping_path) as f:
    mapping = json.load(f)

total  = len(mapping)  # one launch per task
done   = 0
failed = []

for task, groups in mapping.items():
    # Check if all zarrs for this task already exist (skip entire task launch)
    all_done = all(
        os.path.exists(os.path.join(video_dir, "{}_{}.mp4.zarr".format(task, g)))
        for g in groups
    )
    if all_done:
        print("[SKIP] {} — all groups done.".format(task))
        done += 1
        continue

    print("\n[{}/{}] {} — groups {}".format(done + 1, total, task, groups))
    cmd = [
        sys.executable, script,
        "--task",         task,
        "--groups",       *groups,
        "--cameras-file", cameras_file,
        "--image-size",   "256",
        "--fov-deg",      "60.0",
        "--video-only",
        "--video-dir",    video_dir,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("[FAIL] {}: {}".format(task, e))
        failed.append(task)
    done += 1

print("\n[SUMMARY] {}/{} tasks completed.".format(total - len(failed), total))
if failed:
    print("[FAILED]: " + ", ".join(failed))
    sys.exit(1)
PYEOF
