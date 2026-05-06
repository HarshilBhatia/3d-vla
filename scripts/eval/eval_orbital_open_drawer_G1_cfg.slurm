#!/bin/bash

# Usage:
#   sbatch scripts/eval/eval_orbital_open_drawer_G1_cfg.slurm
#   Array index 0→cfg_scale=2, 1→cfg_scale=4, 2→cfg_scale=6
#SBATCH --job-name=eval_orb_cfg
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/eval_orb_G1_cfg_%A_%a.out
#SBATCH --error=logs/eval_orb_G1_cfg_%A_%a.err

set -euo pipefail
ulimit -c 0

echo "Job started at: $(date)"
echo "Node: $(hostname)  Array index: $SLURM_ARRAY_TASK_ID"

HOST_REPO=/home/harshilb/3d_flowmatch_actor
CONTAINER_REPO=/root/3d_flowmatch_actor
CONTAINER=/home/harshilb/my_eval_env
COPPELIASIM_ROOT=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04

CHECKPOINT=/home/harshilb/3d_flowmatch_actor/train_logs/Orbital/open_drawer_multi_groups_clip_langdrop20/step_70000.pth
RUN_NAME=cfg_sweep

CFG_SCALES=(2 4 6)
CFG_SCALE="${CFG_SCALES[$SLURM_ARRAY_TASK_ID]}"

OUTPUT_DIR="${HOST_REPO}/eval_logs/Orbital/${RUN_NAME}/multi_cam_dataset_G1_cfg${CFG_SCALE}/"
OUTPUT_FILE="${OUTPUT_DIR}/results.json"

mkdir -p "$OUTPUT_DIR" logs

echo "Checkpoint: $CHECKPOINT"
echo "CFG scale:  $CFG_SCALE"
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
        checkpoint="$CHECKPOINT" \
        output_file="$OUTPUT_FILE" \
        task="open_drawer" \
        data_dir="/grogu/user/harshilb/open_drawer_multi_cam/" \
        spawn_camera_group="G1" \
        save_video=true \
        cfg_scale="$CFG_SCALE" \
        "$@"

echo "Done at: $(date)"
