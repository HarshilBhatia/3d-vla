#!/usr/bin/env bash
# Standard PerAct2 2D baseline: front + wrist_left + wrist_right RGB views.
# Submit with: sbatch scripts/train/train_peract2_standard_2d_grogu.slurm
#SBATCH --job-name=peract2_standard_2d
#SBATCH --partition=shubhamlong
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/harshilb/3dfa_unified/logs/train/slurm-standard-2d-%j.out
#SBATCH --error=/home/harshilb/3dfa_unified/logs/train/slurm-standard-2d-%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT_OVERRIDE:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
source "${REPO_ROOT}/scripts/helpers/slurm_utils.sh"
# The shared Grogu profile still records the retired 3d_flowmatch_actor path.
# This run is intentionally launched from the current 3dfa_unified checkout.
export CLUSTER_REPO_OVERRIDE="$REPO_ROOT"
load_cluster grogu
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Published standard PerAct2 zarr: cameras are front, wrist_left, wrist_right.
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/grogu/datasets/hbhatia/peract2_standard/Peract2_zarr/train.zarr}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-/grogu/datasets/hbhatia/peract2_standard/Peract2_zarr/val.zarr}"
NGPUS="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
MASTER_PORT=$((27500 + RANDOM % 1000))

for path in "$TRAIN_DATA_DIR" "$EVAL_DATA_DIR"; do
    [[ -d "$path" ]] || { echo "Missing standard PerAct2 zarr: $path"; exit 2; }
done

TRAIN_ARGS=(
    data=full
    experiment=peract2_standard_2d
    dataset=Peract2_3dfront_3dwrist
    train_data_dir="$TRAIN_DATA_DIR"
    eval_data_dir="$EVAL_DATA_DIR"
    exp_log_dir=PerAct2
)
[[ -n "${RUN_LOG_DIR:-}" ]] && TRAIN_ARGS+=("run_log_dir=$RUN_LOG_DIR")

# The standard zarr is NFS-backed. Keep loading in-process by default: a
# multiprocessing worker segfaulted during validation at step 32k.
torchrun --nproc_per_node="$NGPUS" --master_port="$MASTER_PORT" main.py \
    "${TRAIN_ARGS[@]}" \
    num_workers="${NUM_WORKERS:-0}" \
    prefetch_factor="${PREFETCH_FACTOR:-1}" \
    "$@"
