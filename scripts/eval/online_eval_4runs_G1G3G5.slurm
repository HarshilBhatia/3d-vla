#!/bin/bash
# Online eval on open_drawer for 4 checkpoints × 3 camera-group settings.
#   idx 0,1,2   -> cotrain_medium_12_deltaM        on G1, G3, G5
#   idx 3,4,5   -> G3G4_finetune_3dfa              on G1, G3, G5
#   idx 6,7,8   -> G3G4_finetune_deltaM_eef_final  on G1, G3, G5
#   idx 9,10,11 -> 3dfa_multi_cam_G3G4             on G1, G3, G5
# Usage:
#   sbatch scripts/eval/online_eval_4runs_G1G3G5.slurm
#SBATCH --job-name=eval_4runs_G1G3G5
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=4,10
#SBATCH --output=logs/eval_4runs_G1G3G5_%A_%a.out
#SBATCH --error=logs/eval_4runs_G1G3G5_%A_%a.err
#SBATCH --exclude=grogu-0-[14,19,24,29],grogu-1-[6,14],grogu-3-5,grogu-3-25

set -euo pipefail
ulimit -c 0

echo "Job started at: $(date)"
echo "Node: $(hostname)  Array index: $SLURM_ARRAY_TASK_ID"

HOST_REPO=/home/harshilb/3d_flowmatch_actor
CONTAINER_REPO=/root/3d_flowmatch_actor
CONTAINER=/home/harshilb/my_eval_env
COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04

RUN_NAMES=(
    cotrain_medium_12_deltaM
    G3G4_finetune_3dfa
    G3G4_finetune_deltaM_eef_final
    3dfa_multi_cam_G3G4
)
MY_GROUPS=(G1 G3 G5)

RUN_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
GGROUP_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

RUN_NAME="${RUN_NAMES[$RUN_IDX]}"
GGROUP="${MY_GROUPS[$GGROUP_IDX]}"

TASK="open_drawer"
CHECKPOINT="${CONTAINER_REPO}/train_logs/final/${RUN_NAME}/last.pth"
OUTPUT_FILE="${HOST_REPO}/eval_logs/18may_final/${RUN_NAME}/${GGROUP}/results_${TASK}.json"

mkdir -p "$(dirname "$OUTPUT_FILE")" logs

echo "Run:        $RUN_NAME"
echo "GGroup:      $GGROUP"
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
    bash ${CONTAINER_REPO}/scripts/eval/eval_orbital.sh \
        bimanual=false \
        checkpoint="$CHECKPOINT" \
        output_file="$OUTPUT_FILE" \
        task="$TASK" \
        data_dir="/grogu/user/harshilb/low_dim_demos/" \
        camera_groups="GT" \
        spawn_camera_group="$GGROUP" \
        save_video=true \
        "$@"

echo "Done at: $(date)"
