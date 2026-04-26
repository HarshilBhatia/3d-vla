#!/bin/bash
#SBATCH --job-name=strip_pkl
#SBATCH --partition=all
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/strip_pkl_%j.out
#SBATCH --error=logs/strip_pkl_%j.err

set -euo pipefail

REPO_DIR="/home/harshilb/3d_flowmatch_actor"
ROOT="/grogu/user/harshilb/orbital_rollouts"

echo "========================================"
echo "  Job  : ${SLURM_JOB_ID}"
echo "  Node : $(hostname)"
echo "  Start: $(date)"
echo "  Root : ${ROOT}"
echo "========================================"

mkdir -p "${REPO_DIR}/logs"

cd "${REPO_DIR}"
python scripts/strip_pkl.py --root "${ROOT}"

echo "========================================"
echo "  Done: $(date)"
echo "========================================"
