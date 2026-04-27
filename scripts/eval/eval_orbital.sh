#!/usr/bin/env bash
# Evaluate a trained orbital model on RLBench.
# Usage:
#   xvfb-run -a bash scripts/eval/eval_orbital.sh \
#       task=close_jar \
#       checkpoint=train_logs/Orbital/my_run/best.pth \
#       output_file=eval_logs/Orbital/my_run/close_jar.json
#
# Checkpoint-specific arch args (e.g. fps_subsampling_factor=4
# num_vis_instr_attn_layers=2 sa_blocks_use_rope=false) go in "$@".

python online_evaluation_rlbench/evaluate_policy.py \
    val_instructions=instructions/peract/instructions.json \
    data_dir=/grogu/user/harshilb/orbital_rollouts \
    dataset=OrbitalWrist \
    cameras_file=instructions/orbital_cameras_grouped.json \
    task_group_mapping_file=instructions/task_group_mapping_subset.json \
    bimanual=false \
    headless=true \
    max_tries=1 \
    "$@"
