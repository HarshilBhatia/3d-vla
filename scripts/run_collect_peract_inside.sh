#!/usr/bin/env bash
# Runs inside the apptainer container for collect_peract.sbatch.
# All args passed via environment variables set by the sbatch.
set -euo pipefail

export PATH="/root/miniconda3/envs/3dfa/bin:${PATH:-}"
export COPPELIASIM_ROOT="/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"

unset QT_QPA_PLATFORM
export XDG_RUNTIME_DIR=/run/user/27491

exec xvfb-run -a --server-args="-screen 0 1280x1024x24 +extension GLX +render" \
    python /home/harshilb/3d_flowmatch_actor/data/generation/generate.py \
    --tasks             "${COLLECT_TASK}" \
    --save_path         "${COLLECT_SAVE_PATH}" \
    --episodes_per_task "${COLLECT_N_EPISODES}" \
    --offset            "${COLLECT_VAR_OFFSET}" \
    --variations        "${COLLECT_VARS_PER_NODE}" \
    --image_size        128,128 \
    --renderer          opengl3 \
    --processes         1
