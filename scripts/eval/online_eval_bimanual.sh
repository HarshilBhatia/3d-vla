#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# online_eval_bimanual.sh
#
# Run online evaluation across PerAct2 bimanual tasks for a given checkpoint.
# Each task launches CoppeliaSim, evaluates the policy, and writes per-task
# results to output_dir/results_{task}.json.
#
# Usage (from repo root):
#   xvfb-run -a bash scripts/eval/online_eval_bimanual.sh \
#       --checkpoint train_logs/exp/my_run/last.pth \
#       --run-log-dir my_run
#
# Optional overrides:
#   --output-dir   eval_logs/exp/my_run          (default: eval_logs/exp/<run-log-dir>)
#   --tasks        "bimanual_lift_tray ..."       (default: bimanual_lift_tray)
#   --extra        "fps_subsampling_factor=4"     (appended as Hydra overrides)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
CHECKPOINT=""
RUN_LOG_DIR=""
OUTPUT_DIR=""
EXTRA_OVERRIDES=""
TASKS=( bimanual_lift_tray
)

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)   CHECKPOINT="$2";   shift 2 ;;
        --run-log-dir)  RUN_LOG_DIR="$2";  shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --tasks)        IFS=' ' read -r -a TASKS <<< "$2"; shift 2 ;;
        --extra)        EXTRA_OVERRIDES="$2"; shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT" ]]; then
    echo "[ERROR] --checkpoint is required."
    echo "Usage: xvfb-run -a bash scripts/eval/online_eval_bimanual.sh --checkpoint <path> --run-log-dir <name>"
    exit 1
fi
if [[ -z "$RUN_LOG_DIR" ]]; then
    RUN_LOG_DIR="$(basename "$(dirname "$CHECKPOINT")")"
    echo "[INFO] Inferred run-log-dir: $RUN_LOG_DIR"
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${REPO_ROOT}/eval_logs/exp/${RUN_LOG_DIR}"
fi

mkdir -p "$OUTPUT_DIR"
echo "[INFO] checkpoint  : $CHECKPOINT"
echo "[INFO] output_dir  : $OUTPUT_DIR"
echo "[INFO] tasks       : ${TASKS[*]}"
echo

# ── Eval config (overrides that differ from config.yaml / data=full / experiment=default) ──
data_dir=/grogu/user/harshilb/datasets/peract2_raw/peract2_test
max_tries=1     # config default: 10
headless=true

# ── Per-task evaluation loop ──────────────────────────────────────────────────
FAILED_TASKS=()

for task in "${TASKS[@]}"; do
    output_file="${OUTPUT_DIR}/results_${task}.json"

    if [[ -f "$output_file" ]]; then
        echo "[SKIP] $task — results already exist at $output_file"
        continue
    fi

    echo "[EVAL] Starting $task ..."
    t0=$SECONDS

    python "${REPO_ROOT}/online_evaluation_rlbench/evaluate_policy.py" \
        data=full \
        experiment=default \
        data_dir=$data_dir \
        max_tries=$max_tries \
        headless=$headless \
        checkpoint=$CHECKPOINT \
        output_file=$output_file \
        task=$task \
        $EXTRA_OVERRIDES \
    && echo "[DONE] $task in $((SECONDS - t0))s → $output_file" \
    || { echo "[FAIL] $task"; FAILED_TASKS+=("$task"); }

done

# ── Summary ───────────────────────────────────────────────────────────────────
echo
echo "=============================="
echo "  Eval complete: $OUTPUT_DIR"
echo "=============================="

python3 - "$OUTPUT_DIR" <<'EOF'
import json, sys, glob, os

out_dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(out_dir, "results_*.json")))
if not files:
    print("No result files found.")
    sys.exit(0)

means = {}
for f in files:
    with open(f) as fh:
        data = json.load(fh)
    for task, rates in data.items():
        means[task] = rates.get("mean", float("nan"))

col = max(len(t) for t in means)
print(f"\n{'Task':<{col}}  Success")
print("-" * (col + 10))
for t, v in sorted(means.items()):
    print(f"{t:<{col}}  {v:.3f}")
if means:
    overall = sum(means.values()) / len(means)
    print("-" * (col + 10))
    print(f"{'MEAN':<{col}}  {overall:.3f}")
EOF

if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
    echo
    echo "[WARN] Failed tasks: ${FAILED_TASKS[*]}"
    exit 1
fi
