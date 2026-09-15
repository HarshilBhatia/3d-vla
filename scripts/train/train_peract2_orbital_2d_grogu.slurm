#!/usr/bin/env bash
#SBATCH --partition=shubhamlong
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/harshilb/3dfa_unified/logs/train/slurm-2d-%j.out
#SBATCH --error=/home/harshilb/3dfa_unified/logs/train/slurm-2d-%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT_OVERRIDE:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
source "${REPO_ROOT}/scripts/helpers/slurm_utils.sh"
load_cluster grogu
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# This is the same 13-task PerAct2 zarr and 100k-step recipe as the 3D run.
# Override these if the archive is mounted at a different location.
EGRESS_ROOT="${EGRESS_ROOT:-/grogu/datasets/hbhatia/3dfa_egress_extracted/3dfa_egress_20260820}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$EGRESS_ROOT/datasets/orbital_peract2/zarr/train.zarr}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$EGRESS_ROOT/datasets/orbital_peract2/zarr/val.zarr}"
NGPUS="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
MASTER_PORT=$((27500 + RANDOM % 1000))

for path in "$TRAIN_DATA_DIR" "$EVAL_DATA_DIR"; do
    [[ -d "$path" ]] || { echo "Missing zarr path: $path"; exit 2; }
done

TRAIN_ARGS=(
    data=orbital_peract2_nfs
    experiment=peract2_orbital_2d
    dataset=OrbitalPeract2
    train_data_dir="$TRAIN_DATA_DIR"
    eval_data_dir="$EVAL_DATA_DIR"
    exp_log_dir=PerAct2
)
[[ -n "${RUN_LOG_DIR:-}" ]] && TRAIN_ARGS+=("run_log_dir=$RUN_LOG_DIR")

torchrun --nproc_per_node="$NGPUS" --master_port="$MASTER_PORT" main.py \
    "${TRAIN_ARGS[@]}" \
    num_workers="${NUM_WORKERS:-2}" \
    prefetch_factor="${PREFETCH_FACTOR:-1}" \
    "$@"
