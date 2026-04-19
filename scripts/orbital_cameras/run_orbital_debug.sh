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
MAPPING="${REPO_ROOT}/instructions/task_group_mapping.json"
CAMERAS_FILE="${REPO_ROOT}/instructions/orbital_cameras_grouped.json"
VIDEO_DIR="${REPO_ROOT}/debug_videos"
SCRIPT="${REPO_ROOT}/scripts/orbital_cameras/collect.py"

mkdir -p "${VIDEO_DIR}"

if [ ! -f "${MAPPING}" ]; then
    echo "[ERROR] instructions/task_group_mapping.json not found."
    echo "        Run: python scripts/rlbench/create_task_group_mapping.py"
    exit 1
fi


unset DISPLAY
unset QT_QPA_PLATFORM
exec xvfb-run -a python3 - <<PYEOF
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
    done += 1
    for group in groups:
        # Check if this (task, group) zarr already exists
        zarr_path = os.path.join(video_dir, "{}_{}.mp4.zarr".format(task, group))
        if os.path.exists(zarr_path):
            print("[SKIP] {}/{} — already done.".format(task, group))
            continue

        print("\n[{}/{}] {}/{} — one group per launch (avoids OOM)".format(
            done, total, task, group))
        cmd = [
            sys.executable, script,
            "--task",         task,
            "--groups",       group,
            "--cameras-file", cameras_file,
            "--image-size",   "256",
            "--fov-deg",      "60.0",
            "--video-only",
            "--video-dir",    video_dir,
        ]
        env = {k: v for k, v in __import__("os").environ.items() if k != "QT_QPA_PLATFORM"}
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print("[FAIL] {}/{}: {}".format(task, group, e))
            failed.append("{}/{}".format(task, group))

print("\n[SUMMARY] {}/{} tasks completed.".format(total - len(failed), total))
if failed:
    print("[FAILED]: " + ", ".join(failed))
    sys.exit(1)
PYEOF
