#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# online_eval_peract_apptainer.slurm
#
# Run online evaluation for a single task using peract_G1_data inside the
# apptainer container, with GPU support.
#
# Usage:
#   sbatch --export=CHECKPOINT=/path/to/checkpoint.pth \
#          scripts/eval/online_eval_peract_apptainer.slurm
#
# Override task (default: stack_blocks):
#   sbatch --export=CHECKPOINT=...,TASK=close_jar \
#          scripts/eval/online_eval_peract_apptainer.slurm
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=online_eval_peract
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100-32:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/online_eval_peract_%j.out
#SBATCH --error=logs/online_eval_peract_%j.err

set -euo pipefail
ulimit -c 0

echo "Job started at: $(date)"
echo "Node: $(hostname)  Job ID: $SLURM_JOB_ID"

REPO_DIR="/ocean/projects/cis240058p/hbhatia1/3d-vla"
CONTAINER="/ocean/projects/cis240058p/hbhatia1/containers/3dfa-sandbox.sif"
COPPELIASIM_ROOT="${REPO_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
DATA_DIR="${REPO_DIR}/data/peract_G1_data"

: "${CHECKPOINT:?'CHECKPOINT not set — pass via --export=CHECKPOINT=...'}"
TASK="${TASK:-stack_blocks}"

RUN_LOG_DIR="$(basename "$(dirname "$CHECKPOINT")")"
OUTPUT_DIR="${REPO_DIR}/eval_logs/Peract_G1/${RUN_LOG_DIR}"
OUTPUT_FILE="${OUTPUT_DIR}/results_${TASK}.json"

mkdir -p "$OUTPUT_DIR" logs

echo "Task:       $TASK"
echo "Checkpoint: $CHECKPOINT"
echo "Data:       $DATA_DIR"
echo "Output:     $OUTPUT_FILE"

if [[ -f "$OUTPUT_FILE" ]]; then
    echo "[SKIP] Results already exist."
    exit 0
fi

xvfb-run -a \
    apptainer exec \
        --nv \
        --env "COPPELIASIM_ROOT=${COPPELIASIM_ROOT}" \
        --env "LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}" \
        --env "QT_QPA_PLATFORM_PLUGIN_PATH=${COPPELIASIM_ROOT}" \
        --bind "${REPO_DIR}:${REPO_DIR}" \
        "${CONTAINER}" \
    bash scripts/eval/online_eval.sh \
        --checkpoint "$CHECKPOINT" \
        --tasks "$TASK" \
        --output-dir "$OUTPUT_DIR" \
        --extra "eval_data_dir=${DATA_DIR} camera_groups=G1"

echo "Done at: $(date)"
