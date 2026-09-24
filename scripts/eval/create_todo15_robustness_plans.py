#!/usr/bin/env python3
"""Rerun last week's IDLT task-heldout and G7 extreme protocols for the no-view-align (TODO 1) and random-history (TODO 5) checkpoints."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "instructions/eval_plans"
VIDEO = {"visual_num_history": 5, "eval_proprio_history_order": "past_to_current"}
METHODS = [
    {"id": "video_pooled_no_view_align_s140k", "checkpoint": "train_logs/PerAct2/peract2_orbital_vid_deltam_pooled_no_view_align/interm_step_140000.pth", "overrides": VIDEO},
    {"id": "video_pooled_randhist_notime_s050k", "checkpoint": "train_logs/PerAct2/peract2_orbital_vid_deltam_pooled_randhist_notime/interm_step_50000.pth", "overrides": VIDEO},
]
# Campaign ids stay those of last week's runs, so the new method directories sit beside the old ones.
SOURCES = ["idlt_task_heldout_group_v01", "idlt_extreme_task_heldout_group_v01", "idlt_scaled_extreme_task_heldout_group_v01",
           "g7_extreme_opposite_v01", "g7_rotation_only_v01", "g7_opposite_camera_sbrs_v01"]


def write(plan, name):
    out = PLANS / f"{name}_todo15.json"
    out.write_text(json.dumps(plan, indent=2) + "\n")
    cells = sum(len(v) for v in plan["task_calibrations"].values()) * len(plan["methods"])
    print(out.name, plan["campaign_id"], cells, "cells")
    return cells


total = 0
for src in SOURCES:
    plan = json.loads((PLANS / f"{src}.json").read_text())
    plan["methods"] = METHODS
    total += write(plan, src)
# The 30/40 scaled-extreme plan was overwritten by the 50/60 one; rebuild it the same way the generator did.
plan = json.loads((PLANS / "idlt_scaled_extreme_task_heldout_group_v01.json").read_text())
plan["campaign_id"] = "idlt_scaled_extreme_task_heldout_group_v01"
plan["methods"] = METHODS
plan["calibrations"] = ["unknown-e30deg-t30cm", "unknown-e40deg-t40cm"]
plan["task_calibrations"] = {}
for task in plan["tasks"]:
    g = plan["task_viewpoints"][task]["spawn_camera_group"].lower()
    plan["task_calibrations"][task] = [f"seen-{g}-medium-unknown-e30deg-t30cm-v01", f"seen-{g}-medium-unknown-e40deg-t40cm-v01"]
total += write(plan, "idlt_scaled_extreme_3040_task_heldout_group_v01")
print("total cells", total)
