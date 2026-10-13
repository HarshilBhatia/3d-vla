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

ARM="${ARM:?Set ARM=base, deltam_external, miscal_only_base, video_deltam_external, video_deltam_direct, video_deltam_direct_rope3d, video_deltam_refine_rope3d, video_deltam_fullpatch, video_deltam_fullpatch_d8, video_deltam_pooled_no_view_align, video_deltam_pooled_randhist_notime, camframe_rope6d, nowrist4_base, nowrist4_video_pooled, vid_pooled_noalign_randhist_scratch, vid_pooled_noalign_randhist_noreg_from_a50k, or ema20k_vidpooled, ema20k_randhist, ema20k_noviewalign, ema20k_regtest_a, ema20k_regtest_c, ema20k_base, exp1a_adaln_percam, exp1b_adaln_pooled, exp1c_adaln_static, exp2_direct_randhist, exp3_randhist_deltam, exp4a_prenorm, exp4b_inputskip, exp5a_proprio_wrist, exp5b_proprio_none, exp6_fullpatch_direct, exp7a_4task_vel, exp7b_4task_x0, exp7c_4task_rel, exp7d_4task_base_rel, exp7e_4task_base_abs, exp7f_4task_pooled_direct, exp8a_handover_keypose_path, exp8b_handover_chunk8, exp8c_handover_keypose_path_stepfix, exp8d_handover_chunk8_stepfix, exp8e_handover_keypose_segment50, exp9a_lift_ball_chunk8, exp9b_lift_ball_chunk8_noise, exp9c_lift_ball_segment50, exp9d_lift_ball_segment50_noise}"
EXTRA_MODEL_OVERRIDES=()
FROM_SCRATCH=0
NO_PERTURBATION=0
NO_MISCAL=0
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
  video_deltam_pooled_no_view_align)
    EXPERIMENT=peract2_orbital_vid_deltam_pooled_no_view_align
    DEFAULT_RUN_LOG_DIR=peract2_orbital_vid_deltam_pooled_no_view_align
    # Pooled history still feeds camera summaries and the register; only delta_M is off.
    ;;
  video_deltam_pooled_randhist_notime)
    EXPERIMENT=peract2_orbital_vid_deltam_pooled_randhist_notime
    DEFAULT_RUN_LOG_DIR=peract2_orbital_vid_deltam_pooled_randhist_notime
    # Random unordered history set, no time embedding or causal mask; tests temporal use vs extra views.
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
  video_deltam_direct_rope3d)
    EXPERIMENT=peract2_orbital_vid_deltam_direct_fullpatch_rope3d
    DEFAULT_RUN_LOG_DIR=peract2_orbital_vid_deltam_direct_fullpatch_rope3d
    # Experiment owns full_image + patch_rope3d; do not set them by shell override.
    ;;
  video_deltam_refine_rope3d)
    EXPERIMENT=peract2_orbital_vid_deltam_refine_fullpatch_rope3d
    DEFAULT_RUN_LOG_DIR=peract2_orbital_vid_deltam_refine_fullpatch_rope3d
    # Experiment owns full_image + patch_rope3d; never set them by shell override.
    ;;
  video_deltam_fullpatch)
    EXPERIMENT=peract2_orbital_vid_deltam_fullpatch
    DEFAULT_RUN_LOG_DIR=peract2_orbital_vid_deltam_fullpatch_finetune_300k
    # The experiment owns the full-patch architecture and checkpoint-eval
    # cadence. Keep this arm explicit so it cannot accidentally use pooled
    # Video-DeltaM summaries.
    ;;
  video_deltam_fullpatch_d8)
    EXPERIMENT=peract2_orbital_vid_deltam_fullpatch_d8
    DEFAULT_RUN_LOG_DIR=peract2_orbital_vid_deltam_fullpatch_d8
    # Depth 8; warm-start from the pooled refine run, which is the only checkpoint carrying camera_readout.
    ;;
  nowrist4_base)
    EXPERIMENT=peract2_orbital_nowrist4_base
    DEFAULT_RUN_LOG_DIR=peract2_orbital_nowrist4_base
    # External cameras only, four tasks; the experiment config owns dataset and train_tasks.
    FROM_SCRATCH=1
    ;;
  nowrist4_video_pooled)
    EXPERIMENT=peract2_orbital_nowrist4_vid_pooled
    DEFAULT_RUN_LOG_DIR=peract2_orbital_nowrist4_vid_pooled
    # External cameras only, four tasks; same VideoPooled-Refine architecture as the 84k model.
    FROM_SCRATCH=1
    ;;
  vid_pooled_noalign_randhist_scratch)
    EXPERIMENT=peract2_orbital_vid_pooled_noalign_randhist_scratch
    DEFAULT_RUN_LOG_DIR=peract2_orbital_vid_pooled_noalign_randhist_scratch
    # From scratch so no weights were ever trained with delta_M or ordered history.
    FROM_SCRATCH=1
    ;;
  vid_pooled_noalign_randhist_noreg_from_a50k)
    EXPERIMENT=peract2_orbital_vid_pooled_noalign_randhist_noreg_from_a50k
    DEFAULT_RUN_LOG_DIR=peract2_orbital_vid_pooled_noalign_randhist_noreg_from_a50k
    # Twin of vid_pooled_noalign_randhist_scratch sharing its first 50k steps; the register is off from here on.
    DEFAULT_PRETRAINED=/home/harshilb/3dfa_unified/train_logs/PerAct2/peract2_orbital_vid_pooled_noalign_randhist_scratch/interm_step_50000.pth
    ;;
  ema20k_vidpooled)
    EXPERIMENT=peract2_orbital_ema20k_vidpooled
    DEFAULT_RUN_LOG_DIR=peract2_orbital_ema20k_vidpooled
    # 20k EMA fine-tune; the experiment config owns use_ema and the raw/EMA ladders.
    DEFAULT_PRETRAINED=/home/harshilb/3dfa_unified/train_logs/PerAct2/peract2_orbital_video_deltam_external_warmstart_k5v_k3p_a5000_resume/best.pth
    ;;
  ema20k_randhist)
    EXPERIMENT=peract2_orbital_ema20k_randhist
    DEFAULT_RUN_LOG_DIR=peract2_orbital_ema20k_randhist
    # 20k EMA fine-tune; the experiment config owns use_ema and the raw/EMA ladders.
    DEFAULT_PRETRAINED=/home/harshilb/3dfa_unified/train_logs/PerAct2/peract2_orbital_vid_deltam_pooled_randhist_notime/interm_step_50000.pth
    ;;
  ema20k_noviewalign)
    EXPERIMENT=peract2_orbital_ema20k_noviewalign
    DEFAULT_RUN_LOG_DIR=peract2_orbital_ema20k_noviewalign
    # 20k EMA fine-tune; the experiment config owns use_ema and the raw/EMA ladders.
    DEFAULT_PRETRAINED=/home/harshilb/3dfa_unified/train_logs/PerAct2/peract2_orbital_vid_deltam_pooled_no_view_align/interm_step_140000.pth
    ;;
  ema20k_regtest_a)
    EXPERIMENT=peract2_orbital_ema20k_regtest_a
    DEFAULT_RUN_LOG_DIR=peract2_orbital_ema20k_regtest_a
    # 20k EMA fine-tune; the experiment config owns use_ema and the raw/EMA ladders.
    DEFAULT_PRETRAINED=/home/harshilb/3dfa_unified/train_logs/PerAct2/peract2_orbital_vid_pooled_noalign_randhist_scratch/interm_step_160000.pth
    ;;
  ema20k_regtest_c)
    EXPERIMENT=peract2_orbital_ema20k_regtest_c
    DEFAULT_RUN_LOG_DIR=peract2_orbital_ema20k_regtest_c
    # 20k EMA fine-tune; the experiment config owns use_ema and the raw/EMA ladders.
    DEFAULT_PRETRAINED=/home/harshilb/3dfa_unified/train_logs/PerAct2/peract2_orbital_vid_pooled_noalign_randhist_noreg_from_a50k/interm_step_100000.pth
    ;;
  ema20k_base)
    EXPERIMENT=peract2_orbital_ema20k_base
    DEFAULT_RUN_LOG_DIR=peract2_orbital_ema20k_base
    # 20k EMA fine-tune; the experiment config owns use_ema and the raw/EMA ladders.
    DEFAULT_PRETRAINED=/home/harshilb/3dfa_unified/train_logs/PerAct2/peract2_orbital_new_external_only_miscal_base_200k/interm_step_160000.pth
    ;;
  exp1a_adaln_percam)
    EXPERIMENT=peract2_orbital_vid_pooled_randhist_adaln_percam_scratch
    DEFAULT_RUN_LOG_DIR=peract2_orbital_vid_pooled_randhist_adaln_percam_scratch
    # Each camera's evolving C-token adaLN-modulates that camera's tokens in place of delta_M.
    FROM_SCRATCH=1
    ;;
  exp1b_adaln_pooled)
    EXPERIMENT=peract2_orbital_vid_pooled_randhist_adaln_pooled_scratch
    DEFAULT_RUN_LOG_DIR=peract2_orbital_vid_pooled_randhist_adaln_pooled_scratch
    # Mean of the evolving C-tokens adaLN-modulates every token in place of delta_M.
    FROM_SCRATCH=1
    ;;
  exp1c_adaln_static)
    EXPERIMENT=peract2_orbital_vid_pooled_randhist_adaln_static_scratch
    DEFAULT_RUN_LOG_DIR=peract2_orbital_vid_pooled_randhist_adaln_static_scratch
    # Mean of the head-input C-tokens, never updated, adaLN-modulates every token in every layer.
    FROM_SCRATCH=1
    ;;
  exp2_direct_randhist)
    EXPERIMENT=peract2_orbital_vid_direct_randhist_scratch
    DEFAULT_RUN_LOG_DIR=peract2_orbital_vid_direct_randhist_scratch
    # History stack predicts delta_M directly from the current C-tokens; applied to the RoPE.
    FROM_SCRATCH=1
    ;;
  exp3_randhist_deltam)
    EXPERIMENT=peract2_orbital_vid_pooled_randhist_deltam_scratch
    DEFAULT_RUN_LOG_DIR=peract2_orbital_vid_pooled_randhist_deltam_scratch
    # Layer-wise delta_M from the C-tokens on random history; same recipe as the adaLN arms otherwise.
    FROM_SCRATCH=1
    ;;
  exp4a_prenorm)
    EXPERIMENT=peract2_orbital_vid_pooled_randhist_deltam_prenorm_scratch
    DEFAULT_RUN_LOG_DIR=peract2_orbital_vid_pooled_randhist_deltam_prenorm_scratch
    # Exp 3 with a pre-LayerNorm history extractor.
    FROM_SCRATCH=1
    ;;
  exp4b_inputskip)
    EXPERIMENT=peract2_orbital_vid_pooled_randhist_deltam_inputskip_scratch
    DEFAULT_RUN_LOG_DIR=peract2_orbital_vid_pooled_randhist_deltam_inputskip_scratch
    # Exp 3 with the raw current-frame tokens skipped onto the extractor output.
    FROM_SCRATCH=1
    ;;
  exp5a_proprio_wrist)
    EXPERIMENT=peract2_orbital_vid_pooled_randhist_adaln_pooled_proprio_wrist_scratch
    DEFAULT_RUN_LOG_DIR=peract2_orbital_vid_pooled_randhist_adaln_pooled_proprio_wrist_scratch
    # Exp 1b with the proprio encoder reading only the wrist cameras.
    FROM_SCRATCH=1
    ;;
  exp5b_proprio_none)
    EXPERIMENT=peract2_orbital_vid_pooled_randhist_adaln_pooled_proprio_none_scratch
    DEFAULT_RUN_LOG_DIR=peract2_orbital_vid_pooled_randhist_adaln_pooled_proprio_none_scratch
    # Exp 1b with an MLP proprio encoder and no scene context.
    FROM_SCRATCH=1
    ;;
  exp6_fullpatch_direct)
    EXPERIMENT=peract2_orbital_vid_fullpatch_rope3d_randhist_direct_prenorm_inputskip_scratch
    DEFAULT_RUN_LOG_DIR=peract2_orbital_vid_fullpatch_rope3d_randhist_direct_prenorm_inputskip_scratch
    # Full-patch 3D-RoPE history, random frames, pre-LN + input skip, Direct delta_M.
    FROM_SCRATCH=1
    ;;
  exp7a_4task_vel)
    EXPERIMENT=peract2_orbital_4task_fullpatch_direct_vel
    DEFAULT_RUN_LOG_DIR=peract2_orbital_4task_fullpatch_direct_vel
    # Exp 6's architecture on 4 tasks, 100k, velocity prediction.
    TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/orbital_peract2_4task/train.zarr}"
    EVAL_DATA_DIR="${EVAL_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/orbital_peract2_4task/val.zarr}"
    # Dense rows: the 8-record cache window would be 8 near-identical frames.
    CACHE_BATCHES=0
    FROM_SCRATCH=1
    ;;
  exp7b_4task_x0)
    EXPERIMENT=peract2_orbital_4task_fullpatch_direct_x0
    DEFAULT_RUN_LOG_DIR=peract2_orbital_4task_fullpatch_direct_x0
    # Exp 7a with x0 prediction, loss on the implied velocity.
    TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/orbital_peract2_4task/train.zarr}"
    EVAL_DATA_DIR="${EVAL_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/orbital_peract2_4task/val.zarr}"
    # Dense rows: the 8-record cache window would be 8 near-identical frames.
    CACHE_BATCHES=0
    FROM_SCRATCH=1
    ;;
  exp7c_4task_rel)
    EXPERIMENT=peract2_orbital_4task_fullpatch_direct_rel
    DEFAULT_RUN_LOG_DIR=peract2_orbital_4task_fullpatch_direct_rel
    # Exp 7a with relative actions.
    TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/orbital_peract2_4task/train.zarr}"
    EVAL_DATA_DIR="${EVAL_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/orbital_peract2_4task/val.zarr}"
    # Dense rows: the 8-record cache window would be 8 near-identical frames.
    CACHE_BATCHES=0
    FROM_SCRATCH=1
    ;;
  exp7d_4task_base_rel)
    EXPERIMENT=peract2_orbital_4task_base_rel
    DEFAULT_RUN_LOG_DIR=peract2_orbital_4task_base_rel
    # Base on 4 tasks, 100k, relative actions.
    TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/orbital_peract2_4task/train.zarr}"
    EVAL_DATA_DIR="${EVAL_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/orbital_peract2_4task/val.zarr}"
    # Dense rows: the 8-record cache window would be 8 near-identical frames.
    CACHE_BATCHES=0
    FROM_SCRATCH=1
    ;;
  exp7e_4task_base_abs)
    EXPERIMENT=peract2_orbital_4task_base_abs
    DEFAULT_RUN_LOG_DIR=peract2_orbital_4task_base_abs
    # Base on 4 tasks, dense data, absolute actions.
    TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/orbital_peract2_4task/train.zarr}"
    EVAL_DATA_DIR="${EVAL_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/orbital_peract2_4task/val.zarr}"
    # Dense rows: the 8-record cache window would be 8 near-identical frames.
    CACHE_BATCHES=0
    FROM_SCRATCH=1
    ;;
  exp7f_4task_pooled_direct)
    EXPERIMENT=peract2_orbital_4task_pooled_direct
    DEFAULT_RUN_LOG_DIR=peract2_orbital_4task_pooled_direct
    # Exp 2 (pooled Direct delta_M) on 4 tasks, dense data.
    TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/orbital_peract2_4task/train.zarr}"
    EVAL_DATA_DIR="${EVAL_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/orbital_peract2_4task/val.zarr}"
    # Dense rows: the 8-record cache window would be 8 near-identical frames.
    CACHE_BATCHES=0
    FROM_SCRATCH=1
    ;;
  exp8a_handover_keypose_path)
    EXPERIMENT=peract2_orbital_handover_dense_base_keypose_path
    DEFAULT_RUN_LOG_DIR=peract2_orbital_handover_dense_base_keypose_path
    # Base on handover, dense rows, 8 points to the next keypose.
    TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/handover_keypose_path/train.zarr}"
    EVAL_DATA_DIR="${EVAL_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/handover_keypose_path/val.zarr}"
    # Dense rows: the 8-record cache window would be 8 near-identical frames.
    CACHE_BATCHES=0
    FROM_SCRATCH=1
    ;;
  exp8b_handover_chunk8)
    EXPERIMENT=peract2_orbital_handover_dense_base_chunk8
    DEFAULT_RUN_LOG_DIR=peract2_orbital_handover_dense_base_chunk8
    # Base on handover, dense rows, next-8-step chunk.
    TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/handover_chunk8/train.zarr}"
    EVAL_DATA_DIR="${EVAL_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/handover_chunk8/val.zarr}"
    # Dense rows: the 8-record cache window would be 8 near-identical frames.
    CACHE_BATCHES=0
    FROM_SCRATCH=1
    ;;
  exp8c_handover_keypose_path_stepfix)
    EXPERIMENT=peract2_orbital_handover_dense_base_keypose_path_stepfix
    DEFAULT_RUN_LOG_DIR=peract2_orbital_handover_dense_base_keypose_path_stepfix
    # Exp 8a with the trajectory step-embedding fix.
    TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/handover_keypose_path/train.zarr}"
    EVAL_DATA_DIR="${EVAL_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/handover_keypose_path/val.zarr}"
    # Dense rows: the 8-record cache window would be 8 near-identical frames.
    CACHE_BATCHES=0
    FROM_SCRATCH=1
    ;;
  exp8d_handover_chunk8_stepfix)
    EXPERIMENT=peract2_orbital_handover_dense_base_chunk8_stepfix
    DEFAULT_RUN_LOG_DIR=peract2_orbital_handover_dense_base_chunk8_stepfix
    # Exp 8b with the trajectory step-embedding fix.
    TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/handover_chunk8/train.zarr}"
    EVAL_DATA_DIR="${EVAL_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/handover_chunk8/val.zarr}"
    # Dense rows: the 8-record cache window would be 8 near-identical frames.
    CACHE_BATCHES=0
    FROM_SCRATCH=1
    ;;
  exp8e_handover_keypose_segment50)
    EXPERIMENT=peract2_orbital_handover_keypose_segment50_base
    DEFAULT_RUN_LOG_DIR=peract2_orbital_handover_keypose_segment50_base
    # Reference keypose-horizon trajectories on handover keyframe rows.
    TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/handover_keypose_segment50/train.zarr}"
    EVAL_DATA_DIR="${EVAL_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/handover_keypose_segment50/val.zarr}"
    FROM_SCRATCH=1
    ;;
  exp9a_lift_ball_chunk8)
    EXPERIMENT=peract2_lift_ball_dense_base_chunk8
    DEFAULT_RUN_LOG_DIR=peract2_lift_ball_dense_base_chunk8
    # Standard PerAct2 rig: the orbital group-bias table does not apply; any noise is in the experiment config.
    TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/peract2_lift_ball/chunk8/train.zarr}"
    EVAL_DATA_DIR="${EVAL_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/peract2_lift_ball/chunk8/val.zarr}"
    NO_MISCAL=1
    CACHE_BATCHES=0
    FROM_SCRATCH=1
    ;;
  exp9b_lift_ball_chunk8_noise)
    EXPERIMENT=peract2_lift_ball_dense_base_chunk8_noise
    DEFAULT_RUN_LOG_DIR=peract2_lift_ball_dense_base_chunk8_noise
    # Standard PerAct2 rig: the orbital group-bias table does not apply; any noise is in the experiment config.
    TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/peract2_lift_ball/chunk8/train.zarr}"
    EVAL_DATA_DIR="${EVAL_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/peract2_lift_ball/chunk8/val.zarr}"
    NO_MISCAL=1
    CACHE_BATCHES=0
    FROM_SCRATCH=1
    ;;
  exp9c_lift_ball_segment50)
    EXPERIMENT=peract2_lift_ball_keypose_segment50_base
    DEFAULT_RUN_LOG_DIR=peract2_lift_ball_keypose_segment50_base
    # Standard PerAct2 rig: the orbital group-bias table does not apply; any noise is in the experiment config.
    TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/peract2_lift_ball/keypose_segment50/train.zarr}"
    EVAL_DATA_DIR="${EVAL_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/peract2_lift_ball/keypose_segment50/val.zarr}"
    NO_MISCAL=1
    FROM_SCRATCH=1
    ;;
  exp9d_lift_ball_segment50_noise)
    EXPERIMENT=peract2_lift_ball_keypose_segment50_base_noise
    DEFAULT_RUN_LOG_DIR=peract2_lift_ball_keypose_segment50_base_noise
    # Standard PerAct2 rig: the orbital group-bias table does not apply; any noise is in the experiment config.
    TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/peract2_lift_ball/keypose_segment50/train.zarr}"
    EVAL_DATA_DIR="${EVAL_DATA_DIR:-/grogu/datasets/hbhatia/3dfa_dense/peract2_lift_ball/keypose_segment50/val.zarr}"
    NO_MISCAL=1
    FROM_SCRATCH=1
    ;;
  camframe_rope6d)
    EXPERIMENT=peract2_orbital_camframe_rope6d
    DEFAULT_RUN_LOG_DIR=peract2_orbital_camframe_rope6d
    # Camera-frame clouds ignore extrinsics, so no miscalibration or perturbation is passed.
    NO_MISCAL=1
    FROM_SCRATCH=1
    ;;
  *)
    echo "Unknown ARM='$ARM'; expected base, deltam_external, miscal_only_base, video_deltam_external, video_deltam_direct, video_deltam_direct_rope3d, video_deltam_refine_rope3d, video_deltam_fullpatch, video_deltam_fullpatch_d8, video_deltam_pooled_no_view_align, video_deltam_pooled_randhist_notime, camframe_rope6d, nowrist4_base, nowrist4_video_pooled, vid_pooled_noalign_randhist_scratch, vid_pooled_noalign_randhist_noreg_from_a50k, or ema20k_vidpooled, ema20k_randhist, ema20k_noviewalign, ema20k_regtest_a, ema20k_regtest_c, ema20k_base, exp1a_adaln_percam, exp1b_adaln_pooled, exp1c_adaln_static, exp2_direct_randhist, exp3_randhist_deltam, exp4a_prenorm, exp4b_inputskip, exp5a_proprio_wrist, exp5b_proprio_none, exp6_fullpatch_direct, exp7a_4task_vel, exp7b_4task_x0, exp7c_4task_rel, exp7d_4task_base_rel, exp7e_4task_base_abs, exp7f_4task_pooled_direct, exp8a_handover_keypose_path, exp8b_handover_chunk8, exp8c_handover_keypose_path_stepfix, exp8d_handover_chunk8_stepfix, exp8e_handover_keypose_segment50, exp9a_lift_ball_chunk8, exp9b_lift_ball_chunk8_noise, exp9c_lift_ball_segment50, exp9d_lift_ball_segment50_noise" >&2
    exit 2
    ;;
esac

EGRESS_ROOT="${EGRESS_ROOT:-/grogu/datasets/hbhatia/3dfa_egress_extracted/3dfa_egress_20260820}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$EGRESS_ROOT/datasets/orbital_peract2/zarr/train.zarr}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$EGRESS_ROOT/datasets/orbital_peract2/zarr/val.zarr}"
FIXED_NOISE_FILE="${FIXED_NOISE_FILE:-instructions/orbital_miscalibration_noise_bimanual_external_only.json}"
RUN_LOG_DIR="${RUN_LOG_DIR:-$DEFAULT_RUN_LOG_DIR}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${DEFAULT_PRETRAINED:-$REPO_ROOT/train_logs/PerAct2/peract2_orbital_video_deltam_external_20260910/last.pth}}"
# Resume restores model, EMA, optimizer, scheduler position, and iteration.
# Unlike PRETRAINED_CHECKPOINT, it is a true continuation rather than a
# weights-only initialization. On resume the config's train_iters is the *total*.
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
# Any start (requeue, chained continuation, or a twin of a multinode run) continues from this run's own last.pth.
OWN_LAST="$REPO_ROOT/train_logs/PerAct2/$RUN_LOG_DIR/last.pth"
if [[ -z "$RESUME_CHECKPOINT" && -e "$OWN_LAST" ]]; then
    RESUME_CHECKPOINT="$OWN_LAST"
    echo "Resuming from this run's own $RESUME_CHECKPOINT"
fi
# Empty by default: the experiment config is the record of how long a run is.
TRAIN_ITERS="${TRAIN_ITERS:-}"
TRAIN_ITERS_ARGS=(); [[ -n "$TRAIN_ITERS" ]] && TRAIN_ITERS_ARGS=("train_iters=$TRAIN_ITERS")
# Empty by default: the experiment config is the record of the global batch.
BATCH_SIZE="${BATCH_SIZE:-}"
BATCH_SIZE_ARGS=(); [[ -n "$BATCH_SIZE" ]] && BATCH_SIZE_ARGS=("batch_size=$BATCH_SIZE")
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
if [[ "$NO_MISCAL" == "1" ]]; then
    MISCAL_OVERRIDES=()
elif [[ "$NO_PERTURBATION" != "1" ]]; then
    MISCAL_OVERRIDES+=(perturbation_noise_rot_deg=3.0 perturbation_noise_trans_m=0.01)
fi

# Another running job (e.g. a multinode twin of this run) may already own the run directory.
"$REPO_ROOT/scripts/train/claim_run.sh" "$REPO_ROOT/train_logs/PerAct2/$RUN_LOG_DIR" || exit 0

torchrun --nproc_per_node="$NGPUS" --master_port="$MASTER_PORT" main.py \
    data=orbital_peract2_nfs \
    experiment="$EXPERIMENT" \
    train_data_dir="$TRAIN_DATA_DIR" \
    eval_data_dir="$EVAL_DATA_DIR" \
    exp_log_dir=PerAct2 \
    run_log_dir="$RUN_LOG_DIR" \
    "${TRAIN_ITERS_ARGS[@]}" "${BATCH_SIZE_ARGS[@]}" \
    "${MISCAL_OVERRIDES[@]}" \
    video_deltam_cache_batches="${CACHE_BATCHES:-4}" \
    video_deltam_cache_span=8 \
    num_workers="${NUM_WORKERS:-8}" \
    prefetch_factor="${PREFETCH_FACTOR:-2}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${EXTRA_MODEL_OVERRIDES[@]}" \
    "$@"
