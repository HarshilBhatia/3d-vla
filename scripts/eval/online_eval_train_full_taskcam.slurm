#!/bin/bash
# Online eval: 3 train_full ckpts × 3 tasks (task-specific camera groups) × 2 noise conditions.
#
# Array layout (run_idx = idx / 6, task_noise_idx = idx % 6,
#               task_idx = task_noise_idx / 2, noise_idx = task_noise_idx % 2):
#
#   runs:   default (default_3dfa)
#           fixmed  (default_fixed_medium_randnoise)
#           deltaM_eef (deltaM_full_fixed_med_randnoise_EEF)
#
#   task / group / noise:
#     task_noise 0 -> light_bulb_in  G4 / orbital_medium
#     task_noise 1 -> light_bulb_in  G4 / 5deg+5cm
#     task_noise 2 -> place_cups     G1 / orbital_medium
#     task_noise 3 -> place_cups     G1 / 5deg+5cm
#     task_noise 4 -> stack_cups     G4 / orbital_medium
#     task_noise 5 -> stack_cups     G4 / 5deg+5cm
#
#   idx 0  -> default     / light_bulb_in  G4 / orbital_medium
#   idx 1  -> default     / light_bulb_in  G4 / 5deg+5cm
#   idx 2  -> default     / place_cups     G1 / orbital_medium
#   idx 3  -> default     / place_cups     G1 / 5deg+5cm
#   idx 4  -> default     / stack_cups     G4 / orbital_medium
#   idx 5  -> default     / stack_cups     G4 / 5deg+5cm
#   idx 6  -> fixmed      / light_bulb_in  G4 / orbital_medium
#   idx 7  -> fixmed      / light_bulb_in  G4 / 5deg+5cm
#   idx 8  -> fixmed      / place_cups     G1 / orbital_medium
#   idx 9  -> fixmed      / place_cups     G1 / 5deg+5cm
#   idx 10 -> fixmed      / stack_cups     G4 / orbital_medium
#   idx 11 -> fixmed      / stack_cups     G4 / 5deg+5cm
#   idx 12 -> deltaM_eef  / light_bulb_in  G4 / orbital_medium
#   idx 13 -> deltaM_eef  / light_bulb_in  G4 / 5deg+5cm
#   idx 14 -> deltaM_eef  / place_cups     G1 / orbital_medium
#   idx 15 -> deltaM_eef  / place_cups     G1 / 5deg+5cm
#   idx 16 -> deltaM_eef  / stack_cups     G4 / orbital_medium
#   idx 17 -> deltaM_eef  / stack_cups     G4 / 5deg+5cm
#
# Usage:
#   sbatch scripts/eval/online_eval_train_full_taskcam.slurm
#SBATCH --job-name=eval_train_full_taskcam
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/eval_train_full_taskcam_%A_%a.out
#SBATCH --error=logs/eval_train_full_taskcam_%A_%a.err
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,14],grogu-2-22,grogu-3-5,grogu-3-25

set -euo pipefail
ulimit -c 0

echo "Job started at: $(date)"
echo "Node: $(hostname)  Array index: $SLURM_ARRAY_TASK_ID"

HOST_REPO=/home/harshilb/3d_flowmatch_actor
CONTAINER_REPO=/root/3d_flowmatch_actor
CONTAINER=/home/harshilb/my_eval_env
COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04

RUN_LABELS=(default fixmed deltaM_eef)
RUN_DIRS=(default_3dfa default_fixed_medium_randnoise deltaM_full_fixed_med_randnoise_EEF)

TASKS=(light_bulb_in place_cups stack_cups)
CAM_GROUPS=(G4 G1 G4)

# noise_idx 0 = orbital training noise (orbital_miscalibration_noise.json, medium level)
# noise_idx 1 = random miscal noise 5deg + 5cm (random_miscal_noise.json)
NOISE_LABELS=(orbital_medium 5deg5cm)

RUN_IDX=$(( SLURM_ARRAY_TASK_ID / 6 ))
TASK_NOISE_IDX=$(( SLURM_ARRAY_TASK_ID % 6 ))
TASK_IDX=$(( TASK_NOISE_IDX / 2 ))
NOISE_IDX=$(( TASK_NOISE_IDX % 2 ))

RUN_LABEL="${RUN_LABELS[$RUN_IDX]}"
RUN_DIR="${RUN_DIRS[$RUN_IDX]}"
TASK="${TASKS[$TASK_IDX]}"
GGROUP="${CAM_GROUPS[$TASK_IDX]}"
NOISE_LABEL="${NOISE_LABELS[$NOISE_IDX]}"

CHECKPOINT="${CONTAINER_REPO}/train_logs/full_dataset/${RUN_DIR}/last.pth"
OUTPUT_FILE="${HOST_REPO}/eval_logs/23may_train_full_taskcam/${RUN_LABEL}/${GGROUP}/${NOISE_LABEL}/results_${TASK}.json"

mkdir -p "$(dirname "$OUTPUT_FILE")" logs

echo "Run:        $RUN_LABEL  ($RUN_DIR)"
echo "Task:       $TASK"
echo "GGroup:     $GGROUP"
echo "Noise:      $NOISE_LABEL"
echo "Checkpoint: $CHECKPOINT"
echo "Output:     $OUTPUT_FILE"

# Build noise args
NOISE_ARGS=()
if [ "$NOISE_IDX" -eq 0 ]; then
    NOISE_ARGS+=(orbital_miscal_noise_level=medium)
else
    NOISE_ARGS+=(++miscal_rot_level=5deg ++miscal_trans_level=5cm)
fi

apptainer exec \
    --nv --fakeroot --writable \
    --pwd ${CONTAINER_REPO} \
    --bind /usr/bin/xvfb-run:/usr/bin/xvfb-run \
    --bind /grogu/user/harshilb:/grogu/user/harshilb \
    --bind ${HOST_REPO}:${CONTAINER_REPO} \
    --env "PATH=/root/miniconda3/envs/3dfa/bin:$PATH" \
    --env "PYTHONPATH=${CONTAINER_REPO}:${PYTHONPATH:-}" \
    --env "COPPELIASIM_ROOT=${COPPELIASIM_ROOT}" \
    --env "LD_LIBRARY_PATH=${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}" \
    --env "QT_QPA_PLATFORM_PLUGIN_PATH=${COPPELIASIM_ROOT}" \
    --env "XDG_RUNTIME_DIR=/run/user/27491" \
    $CONTAINER \
    xvfb-run -a --server-args="-screen 0 1280x1024x24 +extension GLX +render" \
    python ${CONTAINER_REPO}/online_evaluation_rlbench/evaluate_policy.py \
        dataset=OrbitalWrist \
        data_dir=/grogu/user/harshilb/low_dim_demos \
        cameras_file=instructions/orbital_cameras_grouped.json \
        task_group_mapping_file=instructions/task_group_mapping.json \
        val_instructions=instructions/peract/instructions.json \
        "image_size='256,256'" \
        bimanual=false \
        checkpoint="$CHECKPOINT" \
        output_file="$OUTPUT_FILE" \
        task="$TASK" \
        camera_groups=GT \
        spawn_camera_group="$GGROUP" \
        num_demos=20 \
        save_video=true \
        "${NOISE_ARGS[@]}" \
        "$@"

echo "Done at: $(date)"
