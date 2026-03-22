#!/bin/bash
#SBATCH --job-name=peract_collect
#SBATCH --partition=all
#SBATCH --exclude=grogu-1-34,grogu-0-14,grogu-1-24,grogu-1-14,grogu-2-22
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --export=ALL,PERACT_ROT10_SKIP_PYREP_REBUILD=1
#SBATCH --time=4:00:00
#SBATCH --output=logs/%j-peract_collect_%x.out
#SBATCH --error=logs/%j-peract_collect_%x.err

set -eo pipefail
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND (exit $?)" >&2' ERR

# Disable core dumps -- CoppeliaSim crashes write multi-GB core files to /tmp
ulimit -c 0

: "${TASK:?TASK env var is required}"
: "${CONFIG_NAME:?CONFIG_NAME env var is required}"

REPO_DIR="${REPO_DIR:-/home/lzaceria/mscv/3dvla/3d-vla}"
EPISODES_PER_BATCH="${EPISODES_PER_BATCH:-25}"
BATCH_IDX="${BATCH_IDX:-0}"
EP_OFFSET="${EP_OFFSET:-$(( BATCH_IDX * EPISODES_PER_BATCH ))}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
CAM_ROT_DEG="${CAM_ROT_DEG:-0.0}"
CAM_TRANS_X="${CAM_TRANS_X:-0.0}"
CAM_TRANS_Y="${CAM_TRANS_Y:-0.0}"
CAM_TRANS_Z="${CAM_TRANS_Z:-0.0}"

PERACT_PACKAGED_OUT="${PERACT_PACKAGED_OUT:-$REPO_DIR/Peract_packaged_perturbed}"
SCRATCH_DIR="${TMPDIR:-/tmp}/peract_collect_${SLURM_JOB_ID:-$$}_${TASK}"
RAW_DIR="$SCRATCH_DIR/raw"

cleanup() {
    echo "[INFO] Cleaning up scratch: $SCRATCH_DIR"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

echo "=========================================="
echo "Job started at: $(date)  node=$(hostname)"
echo "  TASK=$TASK  CONFIG=$CONFIG_NAME"
echo "  BATCH_IDX=$BATCH_IDX  EP_OFFSET=$EP_OFFSET"
echo "  CAM_ROT_DEG=$CAM_ROT_DEG"
echo "  CAM_TRANS=($CAM_TRANS_X, $CAM_TRANS_Y, $CAM_TRANS_Z) metres"
echo "  EPISODES_PER_BATCH=$EPISODES_PER_BATCH"
echo "  IMAGE_SIZE=${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  RAW_DIR=$RAW_DIR"
echo "  OUTPUT=$PERACT_PACKAGED_OUT"
echo "=========================================="


cd "$REPO_DIR"
mkdir -p logs "$SCRATCH_DIR" "$RAW_DIR"

CONDA_BASE="${CONDA_BASE:-/home/lzaceria/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate 3dfa

export USER_NAME="${USER_NAME:-LUQMAN}"
export PYTHONPATH="${REPO_DIR}/RLBench:${REPO_DIR}:${PYTHONPATH}"
pip install blosc 2>&1 | tail -1

# CoppeliaSim 18.04
unset COPPELIASIM_ROOT
COPPELIASIM_18="$REPO_DIR/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04"
if [[ -d "$COPPELIASIM_18" ]]; then
  export COPPELIASIM_ROOT="$COPPELIASIM_18"
  echo "Using CoppeliaSim Ubuntu 18.04"
else
  echo "[ERROR] CoppeliaSim not found at $COPPELIASIM_18"; exit 1
fi

if [[ -z "${PERACT_ROT10_SKIP_PYREP_REBUILD}" ]]; then
  pip install cffi >> logs/pyrep_install.log 2>&1
  pip install --no-build-isolation -e "$REPO_DIR/PyRep" >> logs/pyrep_install.log 2>&1 \
    || { echo "[ERROR] PyRep install failed"; cat logs/pyrep_install.log; exit 1; }
fi

# Headless OpenGL via open3d swrast
CONDA_LIB="${CONDA_PREFIX:-/home/lzaceria/miniconda3/envs/3dfa}/lib"
OPEN3D_LIB="$(python -c 'import open3d, os; print(os.path.dirname(open3d.__file__))' 2>/dev/null || true)"
if [[ -n "$OPEN3D_LIB" ]] && [[ -f "$OPEN3D_LIB/swrast_dri.so" ]]; then
  export LIBGL_DRIVERS_PATH="$OPEN3D_LIB"
  export LIBGL_ALWAYS_SOFTWARE=1
  export LD_LIBRARY_PATH="${CONDA_LIB}:${OPEN3D_LIB}:${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="${CONDA_LIB}:${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}"
  echo "[WARN] open3d swrast not found"
fi
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
unset QT_QPA_PLATFORM

if ! command -v xvfb-run &>/dev/null; then
  echo "[ERROR] xvfb-run not found"; exit 1
fi

# Fail fast if /tmp < 8GB -- CoppeliaSim segfaults write multi-GB core dumps, filling /tmp
TMP_AVAIL_KB=$(df /tmp | awk 'NR==2 {print $4}')
TMP_AVAIL_GB=$(( TMP_AVAIL_KB / 1024 / 1024 ))
echo "[INFO] /tmp available: ${TMP_AVAIL_GB}GB  node=$(hostname)"
if [[ $TMP_AVAIL_GB -lt 8 ]]; then
    echo "[ERROR] /tmp only ${TMP_AVAIL_GB}GB free on $(hostname) -- aborting (node needs cleanup)"
    exit 1
fi

# ---- Step 1: Collect demos ----
echo "[1/2] Collecting $EPISODES_PER_BATCH episodes of $TASK (batch=$BATCH_IDX, config=$CONFIG_NAME)..."
generator_rc=0
xvfb-run -a -s "-screen 0 1024x768x24" \
  python RLBench/tools/dataset_generator.py \
    --save_path "$RAW_DIR" \
    --episodes_per_task "$EPISODES_PER_BATCH" \
    --image_size "$IMAGE_SIZE" "$IMAGE_SIZE" \
    --camera_rig_rotation_deg "$CAM_ROT_DEG" \
    --camera_rig_translation "$CAM_TRANS_X","$CAM_TRANS_Y","$CAM_TRANS_Z" \
    --tasks "$TASK" || generator_rc=$?

if [[ $generator_rc -ne 0 ]]; then
  echo "[WARN] dataset_generator exited with code $generator_rc -- converting partial data"
fi

# ---- Step 2: Convert to Peract_packaged ----
echo "[2/2] Converting to Peract_packaged -> $PERACT_PACKAGED_OUT (ep_offset=$EP_OFFSET)..."
python scripts/rlbench/rlbench_to_peract_packaged.py \
  --rlbench_save_path "$RAW_DIR" \
  --out "$PERACT_PACKAGED_OUT" \
  --ep_offset "$EP_OFFSET"

echo "Done at: $(date)"
echo "Output: $PERACT_PACKAGED_OUT/train/${TASK}+*/  (ep${EP_OFFSET}.dat ... ep$(( EP_OFFSET + EPISODES_PER_BATCH - 1 )).dat)"
