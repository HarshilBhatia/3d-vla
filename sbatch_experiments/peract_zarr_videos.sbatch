#!/bin/bash
#SBATCH --job-name=peract_videos
#SBATCH --partition=shubhamlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j-peract_zarr_videos.out
#SBATCH --error=logs/%j-peract_zarr_videos.err

# Produce one verification video from PerAct zarr and one from PerAct rot10 zarr.
# Requires both zarrs to exist (run peract_zarr.sbatch and peract_zarr_rot10_transform.sbatch first).
set -e
REPO_DIR="${REPO_DIR:-/home/lzaceria/mscv/3dvla/3d-vla}"
PERACT_ZARR="${PERACT_ZARR:-$REPO_DIR/Peract_zarr}"
PERACT_ZARR_ROT10="${PERACT_ZARR_ROT10:-$REPO_DIR/Peract_zarr_rot10deg}"
MAX_FRAMES="${MAX_FRAMES:-60}"
CAMERA="${CAMERA:-front}"

echo "Job started at: $(date)"
cd "$REPO_DIR"
source "${CONDA_PREFIX:-/home/lzaceria/miniconda3}/etc/profile.d/conda.sh"
conda activate 3dfa
pip install imageio "imageio[ffmpeg]" -q
mkdir -p logs

# Video from normal PerAct zarr (val.zarr)
if [[ -d "$PERACT_ZARR/val.zarr" ]]; then
  python -m data_processing.zarr_to_video \
    --zarr "$PERACT_ZARR/val.zarr" \
    --out "$PERACT_ZARR/verify_peract.mp4" \
    --camera "$CAMERA" \
    --max_frames "$MAX_FRAMES"
  echo "[OK] PerAct video: $PERACT_ZARR/verify_peract.mp4"
else
  echo "[SKIP] PerAct zarr not found: $PERACT_ZARR/val.zarr"
fi

# Video from PerAct rot10 zarr (val.zarr)
if [[ -d "$PERACT_ZARR_ROT10/val.zarr" ]]; then
  python -m data_processing.zarr_to_video \
    --zarr "$PERACT_ZARR_ROT10/val.zarr" \
    --out "$PERACT_ZARR_ROT10/verify_peract_rot10.mp4" \
    --camera "$CAMERA" \
    --max_frames "$MAX_FRAMES"
  echo "[OK] PerAct rot10 video: $PERACT_ZARR_ROT10/verify_peract_rot10.mp4"
else
  echo "[SKIP] PerAct rot10 zarr not found: $PERACT_ZARR_ROT10/val.zarr"
fi

echo "Job finished at: $(date)"
