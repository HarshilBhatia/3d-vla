#!/usr/bin/env bash
# Generic Slurm entry point for a declarative online-evaluation plan.
# Example:
#   sbatch --array=0-103 scripts/eval/online_eval_plan.slurm \
#     instructions/eval_plans/g7_external_calibration.json
# The array range is the plan's method × viewpoint × calibration × task count.
# This wrapper intentionally has no task, camera, or calibration case logic.

#SBATCH --partition=all
#SBATCH --output=/home/harshilb/3dfa_unified/logs/eval/online_eval_plan-%A_%a.out
#SBATCH --error=/home/harshilb/3dfa_unified/logs/eval/online_eval_plan-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --requeue
# Keep every attempt's log; a requeue would otherwise overwrite the failure that caused it.
#SBATCH --open-mode=append
# grogu-3-20's RTX 2080 Ti is not visible through the evaluation container
# (PyTorch reports CUDA unknown error before any rollout starts); grogu-0-15
# hangs PyRep before producing a rollout artifact.
#SBATCH --exclude=grogu-0-15,grogu-3-20,grogu-3-30,grogu-4-23,grogu-4-3,grogu-4-13

set -euo pipefail
PLAN="${1:?Pass an evaluation plan JSON path}"
REPO_ROOT="${REPO_ROOT_OVERRIDE:-${SLURM_SUBMIT_DIR:-/home/harshilb/3dfa_unified}}"
CELL_INDEX="${SLURM_ARRAY_TASK_ID:?Run this script as a Slurm array}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-/home/harshilb/my_eval_env}"
PYREP_DIR="${PYREP_DIR:-/home/harshilb/peract2_eval_deps/PyRep}"
RLBENCH_DIR="${RLBENCH_DIR:-/home/harshilb/peract2_eval_deps/RLBench}"

cd "$REPO_ROOT"
# A CUDA/container startup failure is node-specific in practice. A failed cell exits
# non-zero (so Slurm shows FAILED) and submits a fresh one-cell retry that excludes
# every node this cell has already failed on. A running job's exclude list cannot be
# changed, and `scontrol requeue` kills the script before it could be, hence a new job.
MAX_RETRIES="${EVAL_MAX_RETRIES:-3}"
RETRY="${EVAL_RETRY:-0}"
FAILED_NODES="${EVAL_FAILED_NODES:-}"

if apptainer exec --nv --fakeroot \
  --bind "$REPO_ROOT:/workspace/3dfa_unified" \
  --bind "$PYREP_DIR:/workspace/PerAct2PyRep" \
  --bind "$RLBENCH_DIR:/workspace/PerAct2RLBench" \
  --bind "/home/harshilb/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04:/opt/CoppeliaSim" \
  --bind "/grogu:/grogu" "$APPTAINER_IMAGE" bash -lc \
  'export PATH=/root/miniconda3/envs/3dfa/bin:/root/miniconda3/bin:$PATH; export PYTHONPATH=/workspace/3dfa_unified:/workspace/PerAct2PyRep:/workspace/PerAct2RLBench; export COPPELIASIM_ROOT=/opt/CoppeliaSim; export LD_LIBRARY_PATH=/opt/CoppeliaSim:${LD_LIBRARY_PATH:-}; export QT_QPA_PLATFORM_PLUGIN_PATH=/opt/CoppeliaSim; unset QT_QPA_PLATFORM; cd /workspace/3dfa_unified; xvfb-run -a /root/miniconda3/envs/3dfa/bin/python scripts/eval/run_eval_plan.py --plan "$1" --cell-index "$2"' _ "$PLAN" "$CELL_INDEX"; then
  exit 0
else
  EXIT_CODE=$?
fi

echo "Evaluation cell ${CELL_INDEX} failed on ${SLURMD_NODENAME} (exit ${EXIT_CODE})." >&2
if (( RETRY < MAX_RETRIES )); then
  FAILED_NODES="${FAILED_NODES:+$FAILED_NODES,}${SLURMD_NODENAME}"
  BASE_EXCLUDE=$(sed -n 's/^#SBATCH --exclude=//p' "$REPO_ROOT/scripts/eval/online_eval_plan.slurm")
  RETRY_JOB=$(EVAL_RETRY=$((RETRY + 1)) EVAL_FAILED_NODES="$FAILED_NODES" sbatch --parsable \
    --array="$CELL_INDEX" --exclude="${BASE_EXCLUDE},${FAILED_NODES}" \
    "$REPO_ROOT/scripts/eval/online_eval_plan.slurm" "$PLAN") \
    && echo "Retry $((RETRY + 1))/${MAX_RETRIES} submitted as ${RETRY_JOB}, excluding ${FAILED_NODES}." >&2 \
    || echo "Could not submit a retry for cell ${CELL_INDEX}." >&2
else
  echo "Cell ${CELL_INDEX} gave up after ${MAX_RETRIES} retries (failed on: ${FAILED_NODES:+$FAILED_NODES,}${SLURMD_NODENAME})." >&2
fi
exit "$EXIT_CODE"
