#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# online_eval.sh
#
# Run online evaluation across all 18 PerAct tasks for a given checkpoint.
# Each task launches CoppeliaSim, evaluates the policy, and writes per-task
# results to output_dir/results_{task}.json.
#
# Usage (from repo root):
#   xvfb-run -a bash scripts/eval/online_eval.sh \
#       --checkpoint train_logs/Orbital/my_run/last.pth \
#       --run-log-dir my_run
#
# Optional overrides:
#   --output-dir   eval_logs/Orbital/my_run   (default: eval_logs/Orbital/<run-log-dir>)
#   --tasks        "close_jar open_drawer"     (default: all 18 PerAct tasks)
#   --extra        "miscalibration_noise_level=small seed=1"  (appended as Hydra overrides)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
CHECKPOINT=""
RUN_LOG_DIR=""
OUTPUT_DIR=""
EXTRA_OVERRIDES=""
TASKS=(
    place_cups close_jar insert_onto_square_peg
    light_bulb_in meat_off_grill open_drawer
    place_shape_in_shape_sorter place_wine_at_rack_location
    push_buttons put_groceries_in_cupboard
    put_item_in_drawer put_money_in_safe reach_and_drag
    slide_block_to_color_target stack_blocks stack_cups
    sweep_to_dustpan_of_size turn_tap
)

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)   CHECKPOINT="$2";   shift 2 ;;
        --run-log-dir)  RUN_LOG_DIR="$2";  shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --tasks)        IFS=' ' read -r -a TASKS <<< "$2"; shift 2 ;;
        --extra)        EXTRA_OVERRIDES="$2"; shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT" ]]; then
    echo "[ERROR] --checkpoint is required."
    echo "Usage: xvfb-run -a bash scripts/eval/online_eval.sh --checkpoint <path> --run-log-dir <name>"
    exit 1
fi
if [[ -z "$RUN_LOG_DIR" ]]; then
    # Infer from checkpoint path: train_logs/Orbital/<run_log_dir>/last.pth
    RUN_LOG_DIR="$(basename "$(dirname "$CHECKPOINT")")"
    echo "[INFO] Inferred run-log-dir: $RUN_LOG_DIR"
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${REPO_ROOT}/eval_logs/Orbital/${RUN_LOG_DIR}"
fi

mkdir -p "$OUTPUT_DIR"
echo "[INFO] checkpoint  : $CHECKPOINT"
echo "[INFO] output_dir  : $OUTPUT_DIR"
echo "[INFO] tasks       : ${TASKS[*]}"
echo

# ── Shared model config (must match checkpoint) ───────────────────────────────
eval_data_dir=/grogu/user/harshilb/orbital_train.zarr
val_instructions=instructions/peract/instructions.json
cameras_file=instructions/orbital_cameras_grouped.json
task_group_mapping_file=instructions/task_group_mapping.json

dataset=OrbitalWrist
image_size="256,256"
fov_deg=60.0
miscalibration_noise_level=null
max_steps=25
prediction_len=1
num_history=3
max_tries=1

model_type=denoise3d
bimanual=false
backbone=clip
finetune_backbone=false
finetune_text_encoder=false
fps_subsampling_factor=4
C=120
num_attn_heads=8
num_vis_instr_attn_layers=2
num_shared_attn_layers=4
relative_action=false
rotation_format=quat_xyzw
denoise_timesteps=5
denoise_model=rectified_flow

learn_extrinsics=false
predict_extrinsics=true
extrinsics_prediction_mode=delta_m_full
dynamic_rope_from_camtoken=true
use_front_camera_frame=false
pc_rotate_by_front_camera=false

traj_scene_rope=true
rope_type=normal
use_com_rope=false
com_rope_block_size=0
com_rope_num_axes=0
com_rope_init_std=0.0
sa_blocks_use_rope=true

headless=true
collision_checking=false
seed=0

# ── Per-task evaluation loop ──────────────────────────────────────────────────
FAILED_TASKS=()

for task in "${TASKS[@]}"; do
    output_file="${OUTPUT_DIR}/results_${task}.json"

    if [[ -f "$output_file" ]]; then
        echo "[SKIP] $task — results already exist at $output_file"
        continue
    fi

    echo "[EVAL] Starting $task ..."
    t0=$SECONDS

    python "${REPO_ROOT}/online_evaluation_rlbench/evaluate_policy.py" \
        val_instructions=$val_instructions \
        eval_data_dir=$eval_data_dir \
        dataset=$dataset \
        cameras_file=$cameras_file \
        task_group_mapping_file=$task_group_mapping_file \
        fov_deg=$fov_deg \
        miscalibration_noise_level=$miscalibration_noise_level \
        "image_size='$image_size'" \
        max_steps=$max_steps \
        prediction_len=$prediction_len \
        num_history=$num_history \
        max_tries=$max_tries \
        model_type=$model_type \
        bimanual=$bimanual \
        backbone=$backbone \
        finetune_backbone=$finetune_backbone \
        finetune_text_encoder=$finetune_text_encoder \
        fps_subsampling_factor=$fps_subsampling_factor \
        embedding_dim=$C \
        num_attn_heads=$num_attn_heads \
        num_vis_instr_attn_layers=$num_vis_instr_attn_layers \
        num_shared_attn_layers=$num_shared_attn_layers \
        relative_action=$relative_action \
        rotation_format=$rotation_format \
        denoise_timesteps=$denoise_timesteps \
        denoise_model=$denoise_model \
        learn_extrinsics=$learn_extrinsics \
        predict_extrinsics=$predict_extrinsics \
        extrinsics_prediction_mode=$extrinsics_prediction_mode \
        dynamic_rope_from_camtoken=$dynamic_rope_from_camtoken \
        use_front_camera_frame=$use_front_camera_frame \
        pc_rotate_by_front_camera=$pc_rotate_by_front_camera \
        traj_scene_rope=$traj_scene_rope \
        rope_type=$rope_type \
        use_com_rope=$use_com_rope \
        com_rope_block_size=$com_rope_block_size \
        com_rope_num_axes=$com_rope_num_axes \
        com_rope_init_std=$com_rope_init_std \
        sa_blocks_use_rope=$sa_blocks_use_rope \
        checkpoint=$CHECKPOINT \
        output_file=$output_file \
        task=$task \
        headless=$headless \
        collision_checking=$collision_checking \
        seed=$seed \
        $EXTRA_OVERRIDES \
    && echo "[DONE] $task in $((SECONDS - t0))s → $output_file" \
    || { echo "[FAIL] $task"; FAILED_TASKS+=("$task"); }

done

# ── Summary ───────────────────────────────────────────────────────────────────
echo
echo "=============================="
echo "  Eval complete: $OUTPUT_DIR"
echo "=============================="

python3 - "$OUTPUT_DIR" <<'EOF'
import json, sys, glob, os

out_dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(out_dir, "results_*.json")))
if not files:
    print("No result files found.")
    sys.exit(0)

means = {}
for f in files:
    with open(f) as fh:
        data = json.load(fh)
    for task, rates in data.items():
        means[task] = rates.get("mean", float("nan"))

col = max(len(t) for t in means)
print(f"\n{'Task':<{col}}  Success")
print("-" * (col + 10))
for t, v in sorted(means.items()):
    print(f"{t:<{col}}  {v:.3f}")
if means:
    overall = sum(means.values()) / len(means)
    print("-" * (col + 10))
    print(f"{'MEAN':<{col}}  {overall:.3f}")
EOF

if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
    echo
    echo "[WARN] Failed tasks: ${FAILED_TASKS[*]}"
    exit 1
fi
