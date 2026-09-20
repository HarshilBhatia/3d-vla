#!/bin/bash
#SBATCH --job-name=g7_scene_pcd
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/home/harshilb/3dfa_unified/logs/eval/g7_scene_pcd_%j.out
#SBATCH --error=/home/harshilb/3dfa_unified/logs/eval/g7_scene_pcd_%j.err

set -euo pipefail
cd /home/harshilb/3dfa_unified
mkdir -p logs/eval
apptainer exec --nv --fakeroot \
  --bind "$PWD:/workspace/3dfa_unified" \
  --bind /home/harshilb/peract2_eval_deps/PyRep:/workspace/PerAct2PyRep \
  --bind /home/harshilb/peract2_eval_deps/RLBench:/workspace/PerAct2RLBench \
  --bind /home/harshilb/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04:/opt/CoppeliaSim \
  --bind /grogu:/grogu /home/harshilb/my_eval_env bash -lc '
    export PATH=/root/miniconda3/envs/3dfa/bin:/root/miniconda3/bin:$PATH
    export PYTHONPATH=/workspace/3dfa_unified:/workspace/PerAct2PyRep:/workspace/PerAct2RLBench
    export COPPELIASIM_ROOT=/opt/CoppeliaSim
    export LD_LIBRARY_PATH=/opt/CoppeliaSim:${LD_LIBRARY_PATH:-}
    export QT_QPA_PLATFORM_PLUGIN_PATH=/opt/CoppeliaSim
    unset QT_QPA_PLATFORM
    cd /workspace/3dfa_unified
    xvfb-run -a python scripts/plot_capture_g7_scene.py --capture
  '
