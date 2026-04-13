#!/bin/bash
#SBATCH --job-name=cam_explore
#SBATCH --partition=shubhamlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cam-explore-%j.out
#SBATCH --error=logs/cam-explore-%j.err

# ──────────────────────────────────────────────────────────────────────────────
# Random camera position explorer for PerAct / PerAct2.
#
# Builds a colored point cloud from a zarr, samples N_RANDOM random camera
# positions around the workspace, renders synthetic views via point-cloud
# projection, and writes a self-contained HTML gallery.
#
# Open <OUT_DIR>/index.html in a browser: click cards to shortlist, then
# click "Copy selected JSON" to get the 6 configs you want to use.
#
# Override any variable at submit time, e.g.:
#   ZARR=Peract_zarr/val.zarr N_RANDOM=500 sbatch explore_camera_positions.sbatch
# ──────────────────────────────────────────────────────────────────────────────

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="${REPO_DIR:-/home/lzaceria/mscv/3dvla/3d-vla}"
ZARR="${ZARR:-$REPO_DIR/Peract_zarr/val.zarr}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/camera_explore}"

# ── Sampling ──────────────────────────────────────────────────────────────────
N_RANDOM="${N_RANDOM:-300}"       # random candidate count
N_FRAMES="${N_FRAMES:-20}"        # zarr frames merged into point cloud
SEED="${SEED:-0}"

# ── Rendering ─────────────────────────────────────────────────────────────────
FOV_DEG="${FOV_DEG:-65}"
IMG_SIZE="${IMG_SIZE:-256}"

# ── Extras ────────────────────────────────────────────────────────────────────
# Add --save_pngs to also write individual PNG files (one per candidate)
EXTRA_FLAGS="${EXTRA_FLAGS:---save_pngs}"

# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "================================================================"
echo "  PerAct camera position explorer"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $(hostname)"
echo "  Started    : $(date)"
echo "  Zarr       : $ZARR"
echo "  Out dir    : $OUT_DIR"
echo "  n_random   : $N_RANDOM   n_frames=$N_FRAMES   seed=$SEED"
echo "  FOV        : ${FOV_DEG}deg   img_size=${IMG_SIZE}x${IMG_SIZE}"
echo "================================================================"

cd "$REPO_DIR"

source /home/lzaceria/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

export USER_NAME="${USER_NAME:-LUQMAN}"
export MPLBACKEND=Agg

mkdir -p "$OUT_DIR" logs

python scripts/rlbench/explore_camera_positions.py \
    --zarr="$ZARR"               \
    --out="$OUT_DIR"             \
    --n_random="$N_RANDOM"       \
    --n_frames="$N_FRAMES"       \
    --seed="$SEED"               \
    --fov_deg="$FOV_DEG"         \
    --img_size="$IMG_SIZE"       \
    $EXTRA_FLAGS

echo ""
echo "================================================================"
echo "  DONE: $(date)"
echo "  Open: $OUT_DIR/summary_structured.png  (then page_001.png …)"
echo "================================================================"
