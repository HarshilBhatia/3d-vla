#!/usr/bin/env bash
# Dense orbital PerAct2 (4 tasks): one array element = one (task, train group, episode block).
#SBATCH --job-name=dense4_collect
#SBATCH --partition=shubhamlong
#SBATCH --exclude=grogu-4-3,grogu-4-13,grogu-3-10,grogu-3-15,grogu-3-20,grogu-3-25,grogu-3-30,grogu-3-5,grogu-4-18,grogu-1-20,grogu-4-23
#SBATCH --array=0-35
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=23:59:00
#SBATCH --output=/home/harshilb/3dfa_unified/logs/data/dense4_collect-%A_%a.out
#SBATCH --error=/home/harshilb/3dfa_unified/logs/data/dense4_collect-%A_%a.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT_OVERRIDE:-/home/harshilb/3dfa_unified}"
OUT_ROOT="${OUT_ROOT:-/grogu/datasets/hbhatia/3dfa_dense/orbital_peract2_4task}"
RAW_ROOT="${RAW_ROOT:-${OUT_ROOT}/raw}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-/home/harshilb/my_eval_env}"
PYREP_DIR="${PYREP_DIR:-/home/harshilb/peract2_eval_deps/PyRep}"
RLBENCH_DIR="${RLBENCH_DIR:-/home/harshilb/peract2_eval_deps/RLBench}"
EPS_PER_BLOCK="${EPS_PER_BLOCK:-10}"
N_BLOCKS="${N_BLOCKS:-3}"

# Shard order (task, group) follows each task's train_groups in peract2_orbital_task_group_mapping.json.
SHARDS=(
    bimanual_handover_item:G1 bimanual_handover_item:G2 bimanual_handover_item:G3
    bimanual_take_tray_out_of_oven:G1 bimanual_take_tray_out_of_oven:G2 bimanual_take_tray_out_of_oven:G3
    bimanual_dual_push_buttons:G3 bimanual_dual_push_buttons:G4 bimanual_dual_push_buttons:G5
    bimanual_pick_laptop:G2 bimanual_pick_laptop:G3 bimanual_pick_laptop:G4
)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD="${SHARDS[$((IDX / N_BLOCKS))]}"
TASK="${SHARD%%:*}"
GROUP="${SHARD##*:}"
EP_START="${EP_START:-$(( (IDX % N_BLOCKS) * EPS_PER_BLOCK ))}"
N_EPISODES="${N_EPISODES:-$EPS_PER_BLOCK}"
mkdir -p "$RAW_ROOT"
echo "[dense4] idx=$IDX task=$TASK group=$GROUP eps=[$EP_START, $((EP_START + N_EPISODES))) host=$(hostname)"

apptainer exec --fakeroot \
    --bind "$REPO_ROOT:/workspace/3dfa_unified" \
    --bind "$PYREP_DIR:/workspace/PerAct2PyRep" \
    --bind "$RLBENCH_DIR:/workspace/PerAct2RLBench" \
    --bind "/home/harshilb/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04:/opt/CoppeliaSim" \
    --bind "/grogu:/grogu" "$APPTAINER_IMAGE" bash -lc \
    'export PATH=/root/miniconda3/envs/3dfa/bin:/root/miniconda3/bin:$PATH; \
     export PYTHONPATH=/workspace/3dfa_unified:/workspace/PerAct2PyRep:/workspace/PerAct2RLBench; \
     export COPPELIASIM_ROOT=/opt/CoppeliaSim; export PYTHONUNBUFFERED=1; \
     export LD_LIBRARY_PATH=/opt/CoppeliaSim:${LD_LIBRARY_PATH:-}; \
     export QT_QPA_PLATFORM_PLUGIN_PATH=/opt/CoppeliaSim; unset QT_QPA_PLATFORM; \
     cd /workspace/3dfa_unified; \
     xvfb-run -a /root/miniconda3/envs/3dfa/bin/python scripts/orbital_cameras/collect_dense.py "$@"' _ \
    --task "$TASK" --groups "$GROUP" --ep-start "$EP_START" --n-episodes "$N_EPISODES" \
    --save-path "$RAW_ROOT" --cameras-file instructions/orbital_cameras_grouped.json
