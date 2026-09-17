#!/usr/bin/env bash
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:25:00
#SBATCH --output=/home/harshilb/3dfa_unified/logs/multinode/mx-%x-%j.out
#SBATCH --error=/home/harshilb/3dfa_unified/logs/multinode/mx-%x-%j.err
set -uo pipefail
ulimit -l unlimited
source /home/harshilb/miniconda3/etc/profile.d/conda.sh && conda activate 3dfa
REPO="${TREE:-/home/harshilb/3dfa_unified}"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export BLOSC_NTHREADS=1
E=/grogu/datasets/hbhatia/3dfa_egress_extracted/3dfa_egress_20260820
ARGS=(
  data=orbital_peract2_nfs experiment=peract2_orbital_2d dataset=OrbitalPeract2
  train_data_dir=$E/datasets/orbital_peract2/zarr/train.zarr
  eval_data_dir=$E/datasets/orbital_peract2/zarr/val.zarr
  exp_log_dir=ResumeTest num_workers=2 prefetch_factor=1
  train_iters=$TOTAL ckpt_interval_steps=10 val_interval_steps=1000000
  interm_ckpt_interval_steps=1000000 batch_size=8
  run_log_dir=$RUN
)
[[ -n "${CKPT:-}" ]] && ARGS+=(checkpoint=$CKPT)
# Exact index audit trail (see resilience/state.py): the only assertion that
# survives this stack's kernel nondeterminism.
export RESILIENCE_INDEX_LOG="$REPO/train_logs/ResumeTest/$RUN/indices"
mkdir -p "$(dirname "$RESILIENCE_INDEX_LOG")"
echo "### tree=$REPO run=$RUN total=$TOTAL ngpu=${NGPU:-1} ckpt=${CKPT:-none} extra=${EXTRA:-none}"
torchrun --nproc_per_node="${NGPU:-1}" --master_port=$((27000 + RANDOM % 900)) \
    main.py "${ARGS[@]}" ${EXTRA:-} "$@"
