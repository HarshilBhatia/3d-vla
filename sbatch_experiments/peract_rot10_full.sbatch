#!/bin/bash
#SBATCH --job-name=peract_rot10_full
#SBATCH --partition=shubhamlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --export=ALL,PERACT_ROT10_SKIP_PYREP_REBUILD=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j-peract_rot10_full.out
#SBATCH --error=logs/%j-peract_rot10_full.err

# Full rot10 pipeline on cluster (no display): collect -> convert -> zarr.
# Raw PNG data is written to node-local /tmp (scratch) to avoid filling NFS home.
# Only the final zarr (~1-2 GB) is written back to home.
#
# Usage:
#   sbatch sbatch_experiments/peract_rot10_full.sbatch
#   EPISODES_PER_TASK=125 TASKS="close_jar,place_cups" sbatch sbatch_experiments/peract_rot10_full.sbatch
#   IMAGE_SIZE=256 sbatch sbatch_experiments/peract_rot10_full.sbatch  # 256 is default

set -eo pipefail
trap 'echo "[ERROR] Script failed at line $LINENO: $BASH_COMMAND (exit $?)" >&2' ERR

REPO_DIR="${REPO_DIR:-/home/lzaceria/mscv/3dvla/3d-vla}"
PERACT_ZARR_ROT10="${PERACT_ZARR_ROT10:-$REPO_DIR/Peract_zarr_rot10deg}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-125}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"

# Raw data goes to node-local scratch; only zarr (~1-2 GB) lands in home.
SCRATCH_DIR="${TMPDIR:-/tmp}/peract_rot10_${SLURM_JOB_ID:-$$}"
PERACT_RAW_ROT10="${PERACT_RAW_ROT10:-$SCRATCH_DIR/peract_raw_rot10}"

# place_cups and close_jar already collected; remaining 16 tasks:
PERACT_18="insert_onto_square_peg,light_bulb_in,meat_off_grill,open_drawer,place_shape_in_shape_sorter,place_wine_at_rack_location,push_buttons,put_groceries_in_cupboard,put_item_in_drawer,put_money_in_safe,reach_and_drag,slide_block_to_color_target,stack_blocks,stack_cups,sweep_to_dustpan_of_size,turn_tap"
TASKS_FLAG="--tasks ${TASKS:-$PERACT_18}"

echo "Job started at: $(date)"
echo "Full rot10 pipeline (headless): collect -> convert -> zarr"
echo "  SCRATCH_DIR=$SCRATCH_DIR"
echo "  PERACT_RAW_ROT10=$PERACT_RAW_ROT10"
echo "  PERACT_ZARR_ROT10=$PERACT_ZARR_ROT10"
echo "  EPISODES_PER_TASK=$EPISODES_PER_TASK"
echo "  IMAGE_SIZE=${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  TASKS=${TASKS:-all 18}"

cd "$REPO_DIR"
mkdir -p logs "$SCRATCH_DIR"

CONDA_BASE="${CONDA_BASE:-/home/lzaceria/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate 3dfa

export USER_NAME="${USER_NAME:-LUQMAN}"
export PYTHONPATH="${REPO_DIR}/RLBench:${REPO_DIR}:${PYTHONPATH}"
pip install blosc 2>&1 | tail -2

# CoppeliaSim 18.04 (avoids GLIBC_2.29)
unset COPPELIASIM_ROOT
COPPELIASIM_18="$REPO_DIR/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04"
COPPELIASIM_20="$REPO_DIR/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
if [[ -d "$COPPELIASIM_18" ]]; then
  export COPPELIASIM_ROOT="$COPPELIASIM_18"
  echo "Using CoppeliaSim Ubuntu 18.04"
elif [[ -d "$COPPELIASIM_20" ]]; then
  echo "[ERROR] Only CoppeliaSim 20.04 found; needs GLIBC_2.29."
  exit 1
fi
if [[ -n "${COPPELIASIM_ROOT}" ]] && [[ -z "${PERACT_ROT10_SKIP_PYREP_REBUILD}" ]]; then
  echo "Installing cffi and PyRep (log: logs/pyrep_install.log) ..."
  pip install cffi >> logs/pyrep_install.log 2>&1
  pip install --no-build-isolation -e "$REPO_DIR/PyRep" >> logs/pyrep_install.log 2>&1 \
    && echo "PyRep install done." \
    || { echo "[ERROR] PyRep install failed. See logs/pyrep_install.log"; cat logs/pyrep_install.log; exit 1; }
fi

# Headless OpenGL via open3d's bundled swrast_dri.so + libLLVM-10.so.1 from conda
CONDA_LIB="${CONDA_PREFIX:-/home/lzaceria/miniconda3/envs/3dfa}/lib"
OPEN3D_LIB="$(python -c 'import open3d, os; print(os.path.dirname(open3d.__file__))' 2>/dev/null || true)"
if [[ -n "$OPEN3D_LIB" ]] && [[ -f "$OPEN3D_LIB/swrast_dri.so" ]]; then
  export LIBGL_DRIVERS_PATH="$OPEN3D_LIB"
  export LIBGL_ALWAYS_SOFTWARE=1
  export LD_LIBRARY_PATH="${CONDA_LIB}:${OPEN3D_LIB}:${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}"
  echo "Using swrast_dri.so from open3d: $OPEN3D_LIB"
  echo "CONDA_LIB (libLLVM-10): $CONDA_LIB"
else
  export LD_LIBRARY_PATH="${CONDA_LIB}:${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}"
  echo "[WARN] open3d swrast not found; GL may fail without hardware."
fi
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
unset QT_QPA_PLATFORM

# ---- Step 1: Collect demos (camera rig 10 deg) via xvfb virtual display ----
echo "[Step 1/3] RLBench dataset_generator (camera_rig_rotation_deg=10, ${IMAGE_SIZE}x${IMAGE_SIZE}) ..."
if ! command -v xvfb-run &>/dev/null; then
  echo "[ERROR] xvfb-run not found. Install: yum install xorg-x11-server-Xvfb"
  exit 1
fi
df -h "$SCRATCH_DIR" || df -h /tmp  # log available scratch space

xvfb-run -a -s "-screen 0 1024x768x24 +iglx" python RLBench/tools/dataset_generator.py \
  --save_path "$PERACT_RAW_ROT10" \
  --episodes_per_task "$EPISODES_PER_TASK" \
  --image_size "$IMAGE_SIZE" "$IMAGE_SIZE" \
  --camera_rig_rotation_deg 10 \
  $TASKS_FLAG

# ---- Step 2: Convert to Peract_packaged ----
echo "[Step 2/3] Convert to Peract_packaged ..."
python scripts/rlbench/rlbench_to_peract_packaged.py \
  --rlbench_save_path "$PERACT_RAW_ROT10" \
  --out "$PERACT_RAW_ROT10/Peract_packaged"

# ---- Step 3: Build zarr (written to home) ----
echo "[Step 3/3] Build zarr -> $PERACT_ZARR_ROT10 ..."
mkdir -p "$PERACT_ZARR_ROT10"
# NOTE: trailing slash on --root is required: peract_to_zarr builds paths as f'{ROOT}{split}/{task}+{var}/'
python -m data_processing.peract_to_zarr \
  --root "$PERACT_RAW_ROT10/Peract_packaged/" \
  --tgt "$PERACT_ZARR_ROT10" \
  --overwrite

# ---- Cleanup scratch (raw PNGs no longer needed once zarr is built) ----
echo "Cleaning up scratch: $SCRATCH_DIR"
rm -rf "$SCRATCH_DIR"

echo "Job finished at: $(date)"
echo "Rot10 zarrs: $PERACT_ZARR_ROT10/train.zarr $PERACT_ZARR_ROT10/val.zarr"
