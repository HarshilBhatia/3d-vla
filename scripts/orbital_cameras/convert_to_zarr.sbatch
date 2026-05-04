#!/bin/bash
#SBATCH --job-name=orbital_to_zarr
#SBATCH --partition=all
#SBATCH --exclude=grogu-0-29,grogu-3-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/orbital_to_zarr_%j.out
#SBATCH --error=logs/orbital_to_zarr_%j.err

set -euo pipefail

ROOT="/grogu/user/harshilb/multi_cam_dataset"
OUT="${ROOT}/train"

echo "========================================"
echo "  Node   : $(hostname)"
echo "  Started: $(date)"
echo "  Root   : ${ROOT}"
echo "  Out    : ${OUT}"
echo "========================================"

mkdir -p /home/harshilb/3d_flowmatch_actor/logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

python /home/harshilb/3d_flowmatch_actor/data/processing/convert_to_zarr/orbital_to_zarr.py \
    --root    "${ROOT}" \
    --out     "${OUT}" \
    --overwrite

echo "========================================"
echo "  Done: $(date)"
echo "  Zarr: ${OUT}"
echo "========================================"
