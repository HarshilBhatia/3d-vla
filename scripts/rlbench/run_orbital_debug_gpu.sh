#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_orbital_debug_gpu.sh
#
# GPU variant of run_orbital_debug.sh.
# Uses EGL headless rendering (no Xvfb) via Apptainer --nv.
#
# Requirements:
#   - Run on a GPU node (Slurm: #SBATCH --gres=gpu:1)
#   - Container sandbox must have NVIDIA bind-mount stubs created
#     (see: touch /usr/bin/nvidia-smi etc. inside sandbox)
#
# Usage (from repo root, inside --nv Apptainer session OR directly via Slurm):
#   bash scripts/rlbench/run_orbital_debug_gpu.sh
#
# Or via Slurm:
#   sbatch --gres=gpu:1 --wrap \
#     "apptainer exec --nv containers/3dfa-sandbox \
#      bash scripts/rlbench/run_orbital_debug_gpu.sh"
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAPPING="${REPO_ROOT}/instructions/task_group_mapping.json"
CAMERAS_FILE="${REPO_ROOT}/instructions/orbital_cameras_grouped.json"
VIDEO_DIR="${REPO_ROOT}/debug_videos"
SCRIPT="${REPO_ROOT}/scripts/rlbench/collect_orbital_rollouts.py"

mkdir -p "${VIDEO_DIR}"

if [ ! -f "${MAPPING}" ]; then
    echo "[ERROR] instructions/task_group_mapping.json not found."
    echo "        Run: python scripts/rlbench/create_task_group_mapping.py"
    exit 1
fi

# Verify GPU is accessible
if ! nvidia-smi &>/dev/null; then
    echo "[ERROR] nvidia-smi not found. Run inside 'apptainer exec --nv' on a GPU node."
    exit 1
fi
echo "[GPU] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# CoppeliaSim's bundled libqeglfs.so depends on the system libQt5EglFSDeviceIntegration.so.5
# which is a different Qt version — causes a fatal version mismatch.
# Use offscreen instead: Qt renders headlessly with no version conflict, while
# CoppeliaSim's own OpenGL context (created independently) still gets GPU acceleration.
export QT_QPA_PLATFORM=offscreen
unset DISPLAY

python3 - <<PYEOF
import json, os, subprocess, sys

repo_root    = "${REPO_ROOT}"
mapping_path = "${MAPPING}"
cameras_file = "${CAMERAS_FILE}"
video_dir    = "${VIDEO_DIR}"
script       = "${SCRIPT}"

with open(mapping_path) as f:
    mapping = json.load(f)

total  = len(mapping)
done   = 0
failed = []

for task, groups in mapping.items():
    done += 1
    for group in groups:
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
        env = {**__import__("os").environ, "QT_QPA_PLATFORM": "offscreen"}
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
