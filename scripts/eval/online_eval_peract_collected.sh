#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TASK="open_drawer"
CHECKPOINT="${REPO_ROOT}/train_logs/Peract/peract_collected/last.pth"
OUTPUT_FILE="${REPO_ROOT}/eval_logs/Peract/peract_collected/results_${TASK}.json"

mkdir -p "$(dirname "$OUTPUT_FILE")" logs

xvfb-run -a python "${REPO_ROOT}/online_evaluation_rlbench/evaluate_policy.py" \
    dataset=PeractCollected \
    data_dir=/grogu/user/harshilb/peract_rollouts \
    val_instructions=instructions/peract/instructions.json \
    "image_size='128,128'" \
    bimanual=false \
    max_tries=3 \
    headless=true \
    checkpoint=$CHECKPOINT \
    output_file=$OUTPUT_FILE \
    task=$TASK
