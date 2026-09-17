#!/usr/bin/env bash
# Matched 3D orbital PerAct2 "new external only miscal setup": fixed group
# bias on orbital/external cameras only + random all-camera top-up.
# Set ARM=base or ARM=deltam_external at submission.
# The two arms differ only in whether DeltaM is enabled for camera IDs [0,1].
#SBATCH --partition=shubhamlong
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/harshilb/3dfa_unified/logs/train/slurm-fixedbias-randall-%j.out
#SBATCH --error=/home/harshilb/3dfa_unified/logs/train/slurm-fixedbias-randall-%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT_OVERRIDE:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
source "${REPO_ROOT}/scripts/helpers/slurm_utils.sh"
export CLUSTER_REPO_OVERRIDE="${CLUSTER_REPO_OVERRIDE:-$REPO_ROOT}"
load_cluster grogu
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

ARM="${ARM:?Set ARM=base, deltam_external, miscal_only_base, video_deltam_external, video_deltam_direct, or video_deltam_fullpatch}"
EXTRA_MODEL_OVERRIDES=()
FROM_SCRATCH=0
NO_PERTURBATION=0
case "$ARM" in
  base)
    EXPERIMENT=peract2_orbital_new_external_only_miscal_base
    DEFAULT_RUN_LOG_DIR=peract2_orbital_new_external_only_miscal_base
    ;;
  deltam_external)
    EXPERIMENT=peract2_orbital_new_external_only_miscal_deltam_external
    DEFAULT_RUN_LOG_DIR=peract2_orbital_new_external_only_miscal_deltam_external
    ;;
  video_deltam_external)
    EXPERIMENT=peract2_orbital_vid_deltam_pooled
    DEFAULT_RUN_LOG_DIR=peract2_orbital_video_deltam_external_warmstart_k5v_k3p
    # The experiment owns the pooled Video-DeltaM architecture, so this arm
    # cannot drift from the fullpatch sibling through shell overrides.
    ;;
  video_deltam_direct)
    EXPERIMENT=peract2_orbital_vid_deltam_direct
    DEFAULT_RUN_LOG_DIR=peract2_orbital_vid_deltam_direct
    ;;
  miscal_only_base)
    EXPERIMENT=peract2_orbital_miscal_only_base
    DEFAULT_RUN_LOG_DIR=peract2_orbital_miscal_only_base
    # Persistent per-group bias only: no perturbation noise on top, and no view
    # alignment (the experiment config owns the latter).
    NO_PERTURBATION=1
    FROM_SCRATCH=1
    ;;
  video_deltam_fullpatch)
    EXPERIMENT=peract2_orbital_vid_deltam_fullpatch
    DEFAULT_RUN_LOG_DIR=peract2_orbital_vid_deltam_fullpatch_finetune_300k
    # The experiment owns the full-patch architecture and checkpoint-eval
    # cadence. Keep this arm explicit so it cannot accidentally use pooled
    # Video-DeltaM summaries.
    ;;
  *)
    echo "Unknown ARM='$ARM'; expected base, deltam_external, miscal_only_base, video_deltam_external, video_deltam_direct, or video_deltam_fullpatch" >&2
    exit 2
    ;;
esac

EGRESS_ROOT="${EGRESS_ROOT:-/grogu/datasets/hbhatia/3dfa_egress_extracted/3dfa_egress_20260820}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$EGRESS_ROOT/datasets/orbital_peract2/zarr/train.zarr}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$EGRESS_ROOT/datasets/orbital_peract2/zarr/val.zarr}"
FIXED_NOISE_FILE="${FIXED_NOISE_FILE:-instructions/orbital_miscalibration_noise_bimanual_external_only.json}"
RUN_LOG_DIR="${RUN_LOG_DIR:-$DEFAULT_RUN_LOG_DIR}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-$REPO_ROOT/train_logs/PerAct2/peract2_orbital_video_deltam_external_20260910/last.pth}"
# Resume restores model, EMA, optimizer, scheduler position, and iteration.
# Unlike PRETRAINED_CHECKPOINT, it is a true continuation rather than a
# weights-only initialization.  Set TRAIN_ITERS to the desired *total* step
# count (e.g. 200000 when resuming a step-100000 checkpoint).
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
TRAIN_ITERS="${TRAIN_ITERS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NGPUS="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
MASTER_PORT=$((27500 + RANDOM % 1000))

for path in "$TRAIN_DATA_DIR" "$EVAL_DATA_DIR" "$FIXED_NOISE_FILE"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 2; }
done
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    [[ -e "$RESUME_CHECKPOINT" ]] || { echo "Missing RESUME_CHECKPOINT: $RESUME_CHECKPOINT" >&2; exit 2; }
    CHECKPOINT_ARGS=("checkpoint=$RESUME_CHECKPOINT")
elif [[ "$FROM_SCRATCH" == "1" ]]; then
    CHECKPOINT_ARGS=()
else
    CHECKPOINT_ARGS=("pretrained_checkpoint=$PRETRAINED_CHECKPOINT")
fi

# T_applied = T_perturbation @ T_group[camera_group]. The fixed table lists all
# four cameras with identity for wrist_left/right, so only orbital_left/right
# carry the bias; miscal_cameras keeps the wrists clean under the perturbation.
# Built here, after FIXED_NOISE_FILE exists.
MISCAL_OVERRIDES=(miscal_mode=group miscal_group_level=medium
                  "miscal_group_file=$FIXED_NOISE_FILE" 'miscal_cameras=[0,1]')
if [[ "$NO_PERTURBATION" != "1" ]]; then
    MISCAL_OVERRIDES+=(perturbation_noise_rot_deg=3.0 perturbation_noise_trans_m=0.01)
fi

torchrun --nproc_per_node="$NGPUS" --master_port="$MASTER_PORT" main.py \
    data=orbital_peract2_nfs \
    experiment="$EXPERIMENT" \
    train_data_dir="$TRAIN_DATA_DIR" \
    eval_data_dir="$EVAL_DATA_DIR" \
    exp_log_dir=PerAct2 \
    run_log_dir="$RUN_LOG_DIR" \
    train_iters="$TRAIN_ITERS" \
    "${MISCAL_OVERRIDES[@]}" \
    video_deltam_cache_batches=4 \
    video_deltam_cache_span=8 \
    batch_size="$BATCH_SIZE" \
    num_workers="${NUM_WORKERS:-8}" \
    prefetch_factor="${PREFETCH_FACTOR:-2}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${EXTRA_MODEL_OVERRIDES[@]}" \
    "$@"
