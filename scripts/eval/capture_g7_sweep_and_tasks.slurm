#!/bin/bash
#SBATCH --job-name=g7_sweep_views
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/home/harshilb/3dfa_unified/logs/eval/g7_sweep_views_%j.out
#SBATCH --error=/home/harshilb/3dfa_unified/logs/eval/g7_sweep_views_%j.err

# Renders the G7 sweep groups on the six SBRS tasks, then all 13 tasks from G1 and G7 as a task gallery.
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
    xvfb-run -a python scripts/capture_g7_sweep_views.py --png \
      --tasks bimanual_pick_laptop bimanual_push_box bimanual_lift_ball bimanual_pick_plate bimanual_straighten_rope bimanual_handover_item_easy
    xvfb-run -a python scripts/capture_g7_sweep_views.py --png \
      --cameras-file instructions/orbital_cameras_grouped.json --groups G1 G7 \
      --out-dir docs/figures/task_gallery \
      --tasks bimanual_push_box bimanual_lift_ball bimanual_dual_push_buttons bimanual_pick_plate bimanual_put_item_in_drawer bimanual_put_bottle_in_fridge bimanual_handover_item bimanual_pick_laptop bimanual_straighten_rope bimanual_sweep_to_dustpan bimanual_lift_tray bimanual_handover_item_easy bimanual_take_tray_out_of_oven
  '
