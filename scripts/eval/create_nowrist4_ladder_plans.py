#!/usr/bin/env python3
"""Checkpoint ladders for the four-task no-wrist runs: the Base and Video ladders with their task list swapped."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Task-seen camera groups, as in instructions/eval_plans/idlt_task_seen_group_v01.json.
TASKS = [{"task": "bimanual_handover_item", "spawn_camera_group": "G1"},
         {"task": "bimanual_take_tray_out_of_oven", "spawn_camera_group": "G1"},
         {"task": "bimanual_sweep_to_dustpan", "spawn_camera_group": "G4"},
         {"task": "bimanual_dual_push_buttons", "spawn_camera_group": "G3"}]
for src, dst in (("sbrs_checkpoint_ladder_v01", "nowrist4_base_checkpoint_ladder_v01"),
                 ("video_deltam_checkpoint_ladder_v01", "nowrist4_video_checkpoint_ladder_v01")):
    plan = json.loads((ROOT / f"instructions/eval_plans/{src}.json").read_text())
    plan["protocol_id"] = dst
    plan["description"] = "No-wrist four-task ladder (external cameras only): " + plan["description"]
    plan["tasks"] = TASKS
    out = ROOT / f"instructions/eval_plans/{dst}.json"
    out.write_text(json.dumps(plan, indent=2) + "\n")
    print(out)
