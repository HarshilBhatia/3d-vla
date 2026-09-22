#!/usr/bin/env python3
"""Create the low-noise IDLT task-heldout plan for the FullPatch Video-DeltaM arm."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS = [
    {"id": "video_fullpatch_best", "checkpoint": "train_logs/PerAct2/peract2_orbital_vid_deltam_fullpatch_finetune_300k_a6000_b64/best.pth", "overrides": {"visual_num_history": 5, "eval_proprio_history_order": "past_to_current"}},
]
CONDITIONS = ("calibrated", "supported-e3deg-t1cm", "unknown-e5deg-t5cm", "unknown-e10deg-t10cm")

plan = json.loads((ROOT / "instructions/eval_plans/idlt_task_heldout_group_v01.json").read_text())
plan["campaign_id"] = "idlt_fullpatch_lownoise_task_heldout_group_v01"
plan["description"] = "IDLT task_heldout_group: FullPatch Video-DeltaM at the calibrated, 3/1, 5/5, and 10/10 conditions the extreme campaigns never covered."
plan["calibration_registry"] = "instructions/eval_calibrations_idlt_v01.json"
plan["methods"] = METHODS
plan["calibrations"] = list(CONDITIONS)
plan["task_calibrations"] = {}
for task in plan["tasks"]:
    group = plan["task_viewpoints"][task]["spawn_camera_group"].lower()
    plan["task_calibrations"][task] = ["calibrated"] + [f"seen-{group}-medium-{condition}-v01" for condition in CONDITIONS[1:]]
out = ROOT / "instructions/eval_plans" / f"{plan['campaign_id']}.json"
out.write_text(json.dumps(plan, indent=2) + "\n")
print(out)
