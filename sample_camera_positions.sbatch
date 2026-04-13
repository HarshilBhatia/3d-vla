#!/bin/bash
#SBATCH --job-name=cam_sample
#SBATCH --partition=shubhamlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cam-sample-%j.out
#SBATCH --error=logs/cam-sample-%j.err

# ──────────────────────────────────────────────────────────────────────────────
# Camera position sampler for PerAct2.
#
# Loads one frame from a PerAct2 zarr, sweeps a grid of global camera-rig
# perturbations (rotations around Z + XYZ translations), renders the resulting
# views + 3D position plots, and writes a self-contained index.html that can
# be opened in any browser.
#
# Customise the variables below, or override at submit time:
#   ZARR=/path/to/val.zarr ROTATE_Z="-30,0,30" sbatch sample_camera_positions.sbatch
# ──────────────────────────────────────────────────────────────────────────────

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="${REPO_DIR:-/home/lzaceria/mscv/3dvla/3d-vla}"
ZARR="${ZARR:-$REPO_DIR/Peract_zarr/val.zarr}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/camera_samples}"

# ── Sampling grid ──────────────────────────────────────────────────────────────
# Rotation around Z (degrees). Use "0" for no rotation.
ROTATE_Z="${ROTATE_Z:--30,-20,-10,0,10,20,30}"

# Translations in metres along X / Y / Z.
TRANSLATE_X="${TRANSLATE_X:-0}"
TRANSLATE_Y="${TRANSLATE_Y:-0}"
TRANSLATE_Z="${TRANSLATE_Z:--0.15,0,0.15}"

# Rotation axis (x / y / z)
ROTATE_AXIS="${ROTATE_AXIS:-z}"

# ── Frame selection ────────────────────────────────────────────────────────────
ROLLOUT="${ROLLOUT:-0}"
FRAME="${FRAME:-0}"

# ── Misc ──────────────────────────────────────────────────────────────────────
# Pass --save_pngs to also write candidate_XXX/views.png alongside the HTML.
EXTRA_FLAGS="${EXTRA_FLAGS:---save_pngs}"

# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "================================================================"
echo "  PerAct2 camera position sampler"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $(hostname)"
echo "  Started  : $(date)"
echo "  Zarr     : $ZARR"
echo "  Out dir  : $OUT_DIR"
echo "  rotate_z : $ROTATE_Z"
echo "  translate: x=$TRANSLATE_X  y=$TRANSLATE_Y  z=$TRANSLATE_Z"
echo "  rollout=$ROLLOUT  frame=$FRAME"
echo "================================================================"

cd "$REPO_DIR"

source /home/lzaceria/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

export USER_NAME="${USER_NAME:-LUQMAN}"
export MPLBACKEND=Agg   # ensure headless matplotlib even without DISPLAY

mkdir -p "$OUT_DIR" logs

python scripts/rlbench/sample_camera_positions.py \
    --zarr="$ZARR"               \
    --out="$OUT_DIR"             \
    --rotate_z="$ROTATE_Z"       \
    --translate_x="$TRANSLATE_X" \
    --translate_y="$TRANSLATE_Y" \
    --translate_z="$TRANSLATE_Z" \
    --rotate_axis="$ROTATE_AXIS" \
    --rollout="$ROLLOUT"         \
    --frame="$FRAME"             \
    $EXTRA_FLAGS

echo ""
echo "================================================================"
echo "  DONE: $(date)"
echo "  Results: $OUT_DIR/index.html"
echo "           $OUT_DIR/candidate_*/views.png"
echo "================================================================"
