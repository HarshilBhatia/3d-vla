#!/usr/bin/env bash
# Legacy reproducibility launcher. It retains historical construction flags and
# output directories; create a registry entry plus JSON campaign plan for new
# evaluation work (see docs/online_eval_naming.md).
#SBATCH --partition=all
# grogu-3-30 sources a broken root fzf shell hook in non-interactive jobs;
# grogu-4-23 has Blackwell (sm_120) GPUs unsupported by the evaluator's pinned
# PyTorch. Both exit before meaningful evaluation can begin.
#SBATCH --exclude=grogu-3-30,grogu-4-23
#SBATCH --array=0-12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --output=/home/harshilb/3dfa_unified/logs/eval/slurm-camera-%A_%a.out
#SBATCH --error=/home/harshilb/3dfa_unified/logs/eval/slurm-camera-%A_%a.err

set -euo pipefail
# Slurm stages the script under /var/spool, so BASH_SOURCE is not a repository
# path in a batch job. Prefer the caller's submit directory (the repo root for
# supported invocations); retain the script-relative fallback for local runs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT_OVERRIDE:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}}"
source "${REPO_ROOT}/scripts/helpers/slurm_utils.sh"
load_cluster grogu
cd "$REPO_ROOT"
# The shared 3dfa conda env has an editable PyRep entry pointing at a stale
# /root path. Keep clean-repo code first, then expose the accessible Grogu
# PerAct2 forks explicitly so the stale finder is never consulted.
export PYTHONPATH="$REPO_ROOT:/home/harshilb/3d_flowmatch_actor/PyRep:/home/harshilb/3d_flowmatch_actor/RLBench"
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/home/harshilb/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
unset QT_QPA_PLATFORM || true
APPTAINER_IMAGE="${APPTAINER_IMAGE:-/home/harshilb/my_eval_env}"
# Restored PerAct2 evaluator dependencies.  Keep these outside the image's
# empty workspace so the editable PyRep CFFI build persists between jobs.
PYREP_DIR="${PYREP_DIR:-/home/harshilb/peract2_eval_deps/PyRep}"
RLBENCH_DIR="${RLBENCH_DIR:-/home/harshilb/peract2_eval_deps/RLBench}"

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the clean 3D .pth}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-/grogu/datasets/hbhatia/peract2_test/peract2_test}"
CAMERAS_FILE="${CAMERAS_FILE:-${REPO_ROOT}/instructions/orbital_cameras_grouped.json}"
ROT="${MISCAL_ROT_LEVEL:-5deg}"
TRANS="${MISCAL_TRANS_LEVEL:-5cm}"
CAMERA_SET="${CAMERA_SET:?Set CAMERA_SET=external or wrist}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_logs/peract2_clean3d_camera_subset/${CAMERA_SET}}"
TASKS=(
    bimanual_push_box bimanual_lift_ball bimanual_dual_push_buttons
    bimanual_pick_plate bimanual_put_item_in_drawer bimanual_put_bottle_in_fridge
    bimanual_handover_item bimanual_pick_laptop bimanual_straighten_rope
    bimanual_sweep_to_dustpan bimanual_lift_tray bimanual_handover_item_easy
    bimanual_take_tray_out_of_oven
)
TASK="${TASK:-${TASKS[${SLURM_ARRAY_TASK_ID:-0}]}}"
GROUP="${SPAWN_CAMERA_GROUP:-$(python - "$REPO_ROOT/instructions/peract2_orbital_task_group_mapping.json" "$TASK" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))['tasks'][sys.argv[2]]['eval_group'])
PY
)}"
if [[ "$CAMERA_SET" == external ]]; then
    INDICES='[0,1]'
    EVAL_PROTOCOL="external_camera_miscalibration"
elif [[ "$CAMERA_SET" == wrist ]]; then
    INDICES='[2,3]'
    EVAL_PROTOCOL="on_robot_camera_miscalibration"
elif [[ "$CAMERA_SET" == all ]]; then
    INDICES='[0,1,2,3]'
    EVAL_PROTOCOL="all_camera_miscalibration"
elif [[ -n "${MISCAL_CAMERA_INDICES:-}" ]]; then
    INDICES="$MISCAL_CAMERA_INDICES"
    EVAL_PROTOCOL="asymmetric_camera_miscalibration"
else
    echo "CAMERA_SET must be external or wrist"; exit 2
fi

[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT"; exit 2; }
[[ -f "$CAMERAS_FILE" ]] || { echo "Missing camera file: $CAMERAS_FILE"; exit 2; }
[[ -d "$PYREP_DIR" ]] || { echo "Missing PyRep dependency: $PYREP_DIR"; exit 2; }
[[ -d "$RLBENCH_DIR" ]] || { echo "Missing RLBench dependency: $RLBENCH_DIR"; exit 2; }

# Evaluation modes are legacy construction plumbing for one effective
# calibration realization. New reports use CALIBRATION_ID only; old directory
# tags stay unchanged so completed results remain reusable.
EVAL_MODE="${MISCAL_MODE:-auto}"
if [[ "$EVAL_MODE" == auto ]]; then
    [[ -n "${ORBITAL_MISCAL_NOISE_LEVEL:-}" ]] && EVAL_MODE="group_sampled" || EVAL_MODE="sampled"
elif [[ "$EVAL_MODE" == clean ]]; then
    EVAL_MODE="calibrated"
elif [[ "$EVAL_MODE" == miscal ]]; then
    [[ -n "${ORBITAL_MISCAL_NOISE_LEVEL:-}" ]] && EVAL_MODE="group_sampled" || EVAL_MODE="sampled"
fi
case "$EVAL_MODE" in
    calibrated|sampled|group|group_sampled) ;;
    *) echo "MISCAL_MODE must be calibrated, sampled, group, or group_sampled"; exit 2 ;;
esac

# Always include the evaluation condition in the output path. Previously,
# 2deg+2cm and 10deg+10cm runs shared one path and silently overwrote results.
if [[ -n "${EVAL_CONDITION_TAG:-}" ]]; then
    CONDITION_TAG="$EVAL_CONDITION_TAG"
elif [[ "$EVAL_MODE" == calibrated ]]; then
    CONDITION_TAG="clean"
elif [[ "$EVAL_MODE" == group ]]; then
    CONDITION_TAG="group_${ORBITAL_MISCAL_NOISE_LEVEL:?group mode needs ORBITAL_MISCAL_NOISE_LEVEL}"
else
    CONDITION_TAG="${ROT}_${TRANS}"
fi
case "$CAMERA_SET" in
    external) CALIBRATION_SCOPE=ext ;;
    wrist) CALIBRATION_SCOPE=robot ;;
    all) CALIBRATION_SCOPE=all ;;
    *) CALIBRATION_SCOPE=explicit ;;
esac
if [[ -n "${EVAL_CALIBRATION_ID:-}" ]]; then
    CALIBRATION_ID="$EVAL_CALIBRATION_ID"
elif [[ "$EVAL_MODE" == calibrated ]]; then
    CALIBRATION_ID="calibrated"
elif [[ "$EVAL_MODE" == group ]]; then
    CALIBRATION_ID="${CALIBRATION_SCOPE}-group-${ORBITAL_MISCAL_NOISE_LEVEL}-v01"
elif [[ "$EVAL_MODE" == group_sampled ]]; then
    CALIBRATION_ID="${CALIBRATION_SCOPE}-group-${ORBITAL_MISCAL_NOISE_LEVEL}-r${ROT}-t${TRANS}-v01"
else
    CALIBRATION_ID="${CALIBRATION_SCOPE}-r${ROT}-t${TRANS}-v01"
fi
# This script selects each task's held-out group from the G1-G6 rotation unless
# SPAWN_CAMERA_GROUP is explicitly overridden. The manifest records that
# distinction so task-held-out results are not confused with fully OOD G7.
VIEWPOINT_REGIME="${EVAL_VIEWPOINT_REGIME:-task_heldout_group}"
RESULT_ROOT="${OUT_ROOT}/${CONDITION_TAG}"

run_eval() {
    local name="$1" indices="$2"
    local output="${RESULT_ROOT}/${name}/results_${TASK}.json"
    mkdir -p "$(dirname "$output")"
    EVAL_ARGS=(
        data=orbital_peract2_nfs dataset=OrbitalPeract2 bimanual=true \
        data_dir="$EVAL_DATA_DIR" eval_data_dir="$EVAL_DATA_DIR" \
        cameras_file="$CAMERAS_FILE" spawn_camera_group="$GROUP" \
        checkpoint="$CHECKPOINT" task="$TASK" headless=true max_tries=1 \
        eval_use_depth2cloud=true output_file="$output" \
        num_demos_total="${NUM_DEMOS_TOTAL:-10}" \
        eval_protocol="$EVAL_PROTOCOL" \
        eval_calibration_id="$CALIBRATION_ID" \
        eval_viewpoint_regime="$VIEWPOINT_REGIME" \
        "++miscal_camera_indices=$indices"
    )
    if [[ "$EVAL_MODE" == group || "$EVAL_MODE" == group_sampled ]]; then
        EVAL_ARGS+=(orbital_miscal_noise_level="$ORBITAL_MISCAL_NOISE_LEVEL")
    fi
    if [[ ( "$EVAL_MODE" == group || "$EVAL_MODE" == group_sampled ) && -n "${ORBITAL_MISCAL_NOISE_FILE:-}" ]]; then
        EVAL_ARGS+=(orbital_miscal_noise_file="$ORBITAL_MISCAL_NOISE_FILE")
    fi
    if [[ "${SAVE_VIDEO:-false}" == true ]]; then
        EVAL_ARGS+=(save_video=true)
    fi
    if [[ "$EVAL_MODE" == sampled || "$EVAL_MODE" == group_sampled ]]; then
        EVAL_ARGS+=(
            miscal_rot_level="$ROT"
            miscal_trans_level="$TRANS"
        )
    fi
    # --writable prevents Apptainer from mounting the host NVIDIA libraries in
    # this image, leaving PyTorch with no visible CUDA device. The evaluator
    # needs no in-container writes.
    apptainer exec --nv --fakeroot \
        --bind "$REPO_ROOT:/workspace/3dfa_unified" \
        --bind "$PYREP_DIR:/workspace/PerAct2PyRep" \
        --bind "$RLBENCH_DIR:/workspace/PerAct2RLBench" \
        --bind "/home/harshilb/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04:/opt/CoppeliaSim" \
        --bind "/grogu:/grogu" "$APPTAINER_IMAGE" bash -lc \
        'export PATH=/root/miniconda3/envs/3dfa/bin:/root/miniconda3/bin:$PATH; \
         export PYTHONPATH=/workspace/3dfa_unified:/workspace/PerAct2PyRep:/workspace/PerAct2RLBench; \
         export COPPELIASIM_ROOT=/opt/CoppeliaSim; \
         export LD_LIBRARY_PATH=/opt/CoppeliaSim:${LD_LIBRARY_PATH:-}; \
         export QT_QPA_PLATFORM_PLUGIN_PATH=/opt/CoppeliaSim; unset QT_QPA_PLATFORM; \
         cd /workspace/3dfa_unified; \
         xvfb-run -a /root/miniconda3/envs/3dfa/bin/python online_evaluation_rlbench/evaluate_policy.py "$@"' _ \
        "${EVAL_ARGS[@]}"
}

# One array task evaluates one benchmark task. Submit this array twice.
run_eval "$CAMERA_SET" "$INDICES"
