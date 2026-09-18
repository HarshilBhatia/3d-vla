#!/usr/bin/env python3
"""Create the fixed 15/15 and 20/20 IDLT plans for all four model arms."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS = [
    {"id": "base_s160k", "checkpoint": "train_logs/PerAct2/peract2_orbital_new_external_only_miscal_base_200k/interm_step_160000.pth"},
    {"id": "deltam_s140k", "checkpoint": "train_logs/PerAct2/peract2_orbital_new_external_only_miscal_deltam_external_200k/interm_step_140000.pth"},
    {"id": "video_pooled_s084k", "checkpoint": "train_logs/PerAct2/peract2_orbital_video_deltam_external_warmstart_k5v_k3p_a5000_resume/best.pth", "overrides": {"visual_num_history": 5, "eval_proprio_history_order": "past_to_current"}},
    {"id": "video_fullpatch_best", "checkpoint": "train_logs/PerAct2/peract2_orbital_vid_deltam_fullpatch_finetune_300k_a6000_b64/best.pth", "overrides": {"visual_num_history": 5, "eval_proprio_history_order": "past_to_current"}},
]

for source, setting in [("idlt_task_seen_group_v01.json", "task_seen_group"), ("idlt_task_heldout_group_v01.json", "task_heldout_group")]:
    plan = json.loads((ROOT / "instructions/eval_plans" / source).read_text())
    plan["campaign_id"] = f"idlt_extreme_{setting}_v01"
    plan["description"] = f"IDLT {setting}: Base, DeltaM, VideoPooled, and FullPatch at fixed 15/15 and 20/20 unknown noise."
    plan["calibration_registry"] = "instructions/eval_calibrations_idlt_extreme_v01.json"
    plan["methods"] = METHODS
    plan["calibrations"] = ["unknown-e15deg-t15cm", "unknown-e20deg-t20cm"]
    plan["task_calibrations"] = {}
    for task in plan["tasks"]:
        group = plan["task_viewpoints"][task]["spawn_camera_group"].lower()
        plan["task_calibrations"][task] = [
            f"seen-{group}-medium-unknown-e15deg-t15cm-v01",
            f"seen-{group}-medium-unknown-e20deg-t20cm-v01",
        ]
    out = ROOT / "instructions/eval_plans" / f"idlt_extreme_{setting}_v01.json"
    out.write_text(json.dumps(plan, indent=2) + "\n")
    print(out)
