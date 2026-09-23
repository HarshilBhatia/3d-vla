#!/usr/bin/env bash
# Submit ladder evals for any new interm checkpoints of this week's runs; the ledger skips ones already submitted.
cd "$(dirname "$0")/../.."
source /home/harshilb/miniconda3/etc/profile.d/conda.sh && conda activate 3dfa
TL=train_logs/PerAct2
run() { PYTHONPATH=$PWD python scripts/eval/checkpoint_ladder.py --checkpoint-dir "$TL/$1" --method-id "$2" --config "instructions/eval_plans/$3.json" --min-step "$4" --submit 2>&1 | grep -E "Submitted batch job|Error" | sed "s/^/[$2] /"; }
run peract2_orbital_vid_deltam_fullpatch_d8 vid-deltaM-fullpatch-d8 vid_deltam_fullpatch_d8_checkpoint_ladder_v01 50000
run peract2_orbital_nowrist4_base nowrist4-base nowrist4_base_checkpoint_ladder_v01 30000
run peract2_orbital_nowrist4_vid_pooled nowrist4-vid-pooled nowrist4_video_checkpoint_ladder_v01 30000
run peract2_orbital_vid_deltam_pooled_no_view_align vid-deltaM-pooled-no-view-align video_deltam_checkpoint_ladder_v01 50000
run peract2_orbital_camframe_rope6d camframe-rope6d sbrs_checkpoint_ladder_v01 50000
run peract2_orbital_vid_deltam_pooled_randhist_notime vid-deltaM-pooled-randhist-notime video_deltam_checkpoint_ladder_v01 30000
