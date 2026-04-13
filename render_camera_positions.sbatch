#!/bin/bash
#SBATCH --job-name=cam_render
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cam-render-%j.out
#SBATCH --error=logs/cam-render-%j.err

# ──────────────────────────────────────────────────────────────────────────────
# Render candidate camera positions in RLBench/CoppeliaSim.
#
# For each candidate the front camera is moved to that position and a proper
# rendered RGB image is captured directly from the simulator.  Each card shows
# N_RESETS renders (different task resets / scene variations) side by side so
# you can judge viability across initial conditions.
#
# Open <OUT_DIR>/index.html, tick your N_SHORTLIST favourites, click
# "Copy selected JSON" to export the configs.
#
# Override at submit time:
#   TASK=push_buttons N_RANDOM=300 sbatch render_camera_positions.sbatch
#   BIMANUAL=true TASK=bimanual_lift_tray sbatch render_camera_positions.sbatch
# ──────────────────────────────────────────────────────────────────────────────

# ── Task / scene ──────────────────────────────────────────────────────────────
TASK="${TASK:-close_jar}"
BIMANUAL="${BIMANUAL:-false}"   # set to "true" for PerAct2 bimanual
VARIATION="${VARIATION:-0}"

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="${REPO_DIR:-/home/lzaceria/mscv/3dvla/3d-vla}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/camera_render}"

# ── Candidate generation ───────────────────────────────────────────────────────
N_RANDOM="${N_RANDOM:-200}"
N_RESETS="${N_RESETS:-3}"     # scene resets per candidate (more = richer gallery)
SEED="${SEED:-0}"

# ── Optional: use JSON shortlist from explore_camera_positions.py ─────────────
# CANDIDATES_JSON="$REPO_DIR/camera_explore/shortlist.json"
CANDIDATES_JSON="${CANDIDATES_JSON:-}"

# ── Misc ──────────────────────────────────────────────────────────────────────
IMAGE_SIZE="${IMAGE_SIZE:-256}"
EXTRA_FLAGS="${EXTRA_FLAGS:---save_pngs}"

# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "================================================================"
echo "  RLBench camera position renderer"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $(hostname)"
echo "  Started    : $(date)"
echo "  Task       : $TASK  (bimanual=$BIMANUAL)"
echo "  Out dir    : $OUT_DIR"
echo "  n_random=$N_RANDOM  n_resets=$N_RESETS  seed=$SEED"
echo "================================================================"

cd "$REPO_DIR"

source /home/lzaceria/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

export USER_NAME="${USER_NAME:-LUQMAN}"
export PYTHONPATH="${REPO_DIR}/RLBench:${REPO_DIR}:${PYTHONPATH:-}"

# ── CoppeliaSim (Ubuntu 18.04 – compatible with RHEL 8.6 / glibc 2.28) ───────
unset COPPELIASIM_ROOT
COPPELIASIM_18="$REPO_DIR/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04"
if [[ -d "$COPPELIASIM_18" ]]; then
    export COPPELIASIM_ROOT="$COPPELIASIM_18"
    echo "  CoppeliaSim: $COPPELIASIM_ROOT"
else
    echo "[ERROR] CoppeliaSim 18.04 not found at $COPPELIASIM_18"; exit 1
fi

# ── Headless OpenGL: open3d's bundled swrast_dri.so (software Mesa) ──────────
CONDA_LIB="${CONDA_PREFIX:-/home/lzaceria/miniconda3/envs/3dfa}/lib"
OPEN3D_LIB="$(python -c 'import open3d, os; print(os.path.dirname(open3d.__file__))' 2>/dev/null || true)"
if [[ -n "$OPEN3D_LIB" ]] && [[ -f "$OPEN3D_LIB/swrast_dri.so" ]]; then
    export LIBGL_DRIVERS_PATH="$OPEN3D_LIB"
    export LIBGL_ALWAYS_SOFTWARE=1
    export LD_LIBRARY_PATH="${CONDA_LIB}:${OPEN3D_LIB}:${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
    echo "  swrast_dri.so: $OPEN3D_LIB"
else
    export LD_LIBRARY_PATH="${CONDA_LIB}:${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
    echo "  [WARN] open3d swrast not found; GL may fail without hardware GPU"
fi
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
unset QT_QPA_PLATFORM
export MPLBACKEND=Agg

mkdir -p "$OUT_DIR" logs

# ── Build argument list ────────────────────────────────────────────────────────
ARGS=(
    --task="$TASK"
    --out="$OUT_DIR"
    --n_random="$N_RANDOM"
    --n_resets="$N_RESETS"
    --seed="$SEED"
    --variation="$VARIATION"
    --image_size="$IMAGE_SIZE"
)

[[ "$BIMANUAL" == "true" ]] && ARGS+=(--bimanual)
[[ -n "$CANDIDATES_JSON" ]] && ARGS+=(--candidates_json="$CANDIDATES_JSON")

# ── Run under virtual display ──────────────────────────────────────────────────
xvfb-run -a -s "-screen 0 1024x768x24 +iglx" python scripts/rlbench/render_camera_positions.py \
    "${ARGS[@]}" $EXTRA_FLAGS

echo ""
echo "================================================================"
echo "  DONE: $(date)"
  echo "  Open: $OUT_DIR/summary_structured.png  (then page_001.png …)"
echo "================================================================"
