#!/bin/bash
# Online eval: turn_tap × 3 checkpoints × 5 miscal noise levels (rot+trans sweep).
# Array index mapping: RUN_IDX = index / 5, NOISE_IDX = index % 5
# Runs: default_3dfa, default_fixed_medium_randnoise, deltaM_full_fixed_med_randnoise_EEF
# Noise levels: 2, 5, 10, 15, 20 deg / cm

#SBATCH --job-name=eval_miscal_sweep
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/eval_miscal_sweep_%A_%a.out
#SBATCH --error=logs/eval_miscal_sweep_%A_%a.err
#SBATCH --exclude=grogu-2-22

set -euo pipefail
ulimit -c 0

echo "Job started at: $(date)"
echo "Node: $(hostname)  Array index: $SLURM_ARRAY_TASK_ID"

HOST_REPO=/home/harshilb/3d_flowmatch_actor
CONTAINER_REPO=/root/3d_flowmatch_actor
CONTAINER=/home/harshilb/my_eval_env
COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04

RUNS=(
    default_3dfa
    default_fixed_medium_randnoise
    deltaM_full_fixed_med_randnoise_EEF
)
NOISE_LEVELS=(2 5 10 15 20)

RUN_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
NOISE_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))

RUN="${RUNS[$RUN_IDX]}"
LEVEL="${NOISE_LEVELS[$NOISE_IDX]}"
ROT_LEVEL="${LEVEL}deg"
TRANS_LEVEL="${LEVEL}cm"

CHECKPOINT="${CONTAINER_REPO}/train_logs/full_dataset/${RUN}/last.pth"
OUTPUT_FILE="${HOST_REPO}/eval_logs/full_dataset/${RUN}/results_turn_tap_noise${LEVEL}.json"

mkdir -p "$(dirname "$OUTPUT_FILE")" logs

echo "Run:        $RUN"
echo "Noise:      rot=${ROT_LEVEL}  trans=${TRANS_LEVEL}"
echo "Checkpoint: $CHECKPOINT"
echo "Output:     $OUTPUT_FILE"

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
        checkpoint="$CHECKPOINT" \
        output_file="$OUTPUT_FILE" \
        task=turn_tap \
        bimanual=false \
        camera_groups=GT \
        spawn_camera_group=G7 \
        +miscal_rot_level=$ROT_LEVEL \
        +miscal_trans_level=$TRANS_LEVEL \
        save_video=false \
        "$@"

echo "Done at: $(date)"
