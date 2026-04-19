#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_orbital_datagen_gpu.sh
#
# GPU variant of run_orbital_datagen.sh.
# Uses EGL headless rendering (no Xvfb) via Apptainer --nv.
#
# Requirements:
#   - Run on a GPU node (Slurm: #SBATCH --gres=gpu:1)
#   - Container sandbox must have NVIDIA bind-mount stubs created
#     (see: touch /usr/bin/nvidia-smi etc. inside sandbox)
#
# Usage (from repo root):
#   bash scripts/rlbench/run_orbital_datagen_gpu.sh
#
# Override episodes per group:
#   N_EPISODES=5 bash scripts/rlbench/run_orbital_datagen_gpu.sh
#
# Via Slurm:
#   sbatch --gres=gpu:1 --wrap \
#     "apptainer exec --nv containers/3dfa-sandbox \
#      bash scripts/rlbench/run_orbital_datagen_gpu.sh"
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAPPING="${REPO_ROOT}/instructions/task_group_mapping.json"
CAMERAS_FILE="${REPO_ROOT}/instructions/orbital_cameras_grouped.json"
SAVE_PATH="${REPO_ROOT}/data/orbital_rollouts"
SCRIPT="${REPO_ROOT}/scripts/rlbench/collect_orbital_rollouts.py"
N_EPISODES="${N_EPISODES:-30}"

mkdir -p "${SAVE_PATH}"

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

export QT_QPA_PLATFORM=eglfs
unset DISPLAY

# Ensure NVIDIA EGL libs are on the library path (--nv binds them but may not register with ldconfig)
_nvidia_egl=$(find /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu -name "libEGL_nvidia.so*" 2>/dev/null | head -1)
if [ -n "$_nvidia_egl" ]; then
    export LD_LIBRARY_PATH="$(dirname "$_nvidia_egl"):${LD_LIBRARY_PATH:-}"
    echo "[GPU] NVIDIA EGL found at: $_nvidia_egl"
else
    echo "[WARN] libEGL_nvidia.so not found — eglfs may fail"
fi

python3 - <<PYEOF
import json, os, subprocess, sys

repo_root    = "${REPO_ROOT}"
mapping_path = "${MAPPING}"
cameras_file = "${CAMERAS_FILE}"
save_path    = "${SAVE_PATH}"
script       = "${SCRIPT}"
n_episodes   = int("${N_EPISODES}")

with open(mapping_path) as f:
    mapping = json.load(f)

total  = len(mapping)
done   = 0
failed = []

for task, groups in mapping.items():
    print("\n[{}/{}] {} — groups {} ({} eps each)".format(
        done + 1, total, task, groups, n_episodes))
    cmd = [
        sys.executable, script,
        "--task",         task,
        "--groups",       *groups,
        "--n-episodes",   str(n_episodes),
        "--save-path",    save_path,
        "--cameras-file", cameras_file,
        "--image-size",   "256",
        "--fov-deg",      "60.0",
    ]
    env = {**__import__("os").environ, "QT_QPA_PLATFORM": "eglfs"}
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print("[FAIL] {}: {}".format(task, e))
        failed.append(task)
    done += 1

print("\n[SUMMARY] {}/{} tasks completed.".format(total - len(failed), total))
if failed:
    print("[FAILED]: " + ", ".join(failed))
    sys.exit(1)
PYEOF
