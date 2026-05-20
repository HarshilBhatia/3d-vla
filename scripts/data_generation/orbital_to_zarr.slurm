#!/bin/bash
#SBATCH --job-name=orbital_to_zarr
#SBATCH --partition=shubhamlong
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/orbital_to_zarr_%j.out
#SBATCH --error=logs/orbital_to_zarr_%j.err

set -euo pipefail
ulimit -c 0

HOST_REPO=/home/harshilb/3d_flowmatch_actor
CONTAINER_REPO=/root/3d_flowmatch_actor
CONTAINER=/home/harshilb/my_eval_env
COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04

ROOT=/grogu/user/harshilb/full_rollouts_merged
OUT=/grogu/datasets/hbhatia/peract_orb_new

mkdir -p "${HOST_REPO}/logs" "${OUT}"

echo "========================================"
echo "  Job      : ${SLURM_JOB_ID}"
echo "  Node     : $(hostname)"
echo "  Started  : $(date)"
echo "  Root     : ${ROOT}"
echo "  Out      : ${OUT}"
echo "========================================"

apptainer exec \
    --fakeroot --writable \
    --pwd ${CONTAINER_REPO} \
    --bind /usr/bin/xvfb-run:/usr/bin/xvfb-run \
    --bind /grogu/user/harshilb:/grogu/user/harshilb \
    --bind /grogu/datasets/hbhatia:/grogu/datasets/hbhatia \
    --bind ${HOST_REPO}:${CONTAINER_REPO} \
    --env "PATH=/root/miniconda3/envs/3dfa/bin:$PATH" \
    --env "PYTHONPATH=${CONTAINER_REPO}:${PYTHONPATH:-}" \
    --env "COPPELIASIM_ROOT=${COPPELIASIM_ROOT}" \
    --env "LD_LIBRARY_PATH=${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}" \
    --env "QT_QPA_PLATFORM_PLUGIN_PATH=${COPPELIASIM_ROOT}" \
    --env "XDG_RUNTIME_DIR=/run/user/27491" \
    $CONTAINER \
    xvfb-run -a --server-args="-screen 0 1280x1024x24 +extension GLX +render" \
    python data/processing/convert_to_zarr/orbital_to_zarr.py \
        --root "${ROOT}" \
        --out  "${OUT}" \
        --train-episodes 33 \
        --overwrite

echo "========================================"
echo "  Finished : $(date)"
echo "  Output   : ${OUT}"
echo "========================================"
