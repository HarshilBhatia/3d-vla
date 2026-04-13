#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_orbital_datagen.sh
#
# Full rollout generation: 18 tasks × 3 groups × 30 episodes = 1620 episodes.
# One CoppeliaSim launch per task — all groups collected in that session.
#
# Episodes saved to: data/orbital_rollouts/{task}/{group}/episode_{N}/
#
# Usage (from repo root):
#   xvfb-run -a bash scripts/rlbench/run_orbital_datagen.sh
#
# Override episodes per group:
#   N_EPISODES=5 xvfb-run -a bash scripts/rlbench/run_orbital_datagen.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAPPING="${REPO_ROOT}/task_group_mapping.json"
CAMERAS_FILE="${REPO_ROOT}/orbital_cameras_grouped.json"
SAVE_PATH="${REPO_ROOT}/data/orbital_rollouts"
SCRIPT="${REPO_ROOT}/scripts/rlbench/collect_orbital_rollouts.py"
N_EPISODES="${N_EPISODES:-30}"

mkdir -p "${SAVE_PATH}"

if [ ! -f "${MAPPING}" ]; then
    echo "[ERROR] task_group_mapping.json not found."
    echo "        Run: python scripts/rlbench/create_task_group_mapping.py"
    exit 1
fi

unset DISPLAY
exec xvfb-run -a python3 - <<PYEOF
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
