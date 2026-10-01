#!/usr/bin/env python3
"""IDLT task-heldout calibrated + 10/10 → 60/60 for the no-wrist 4-task models, on the two tasks they solve (sweep, oven)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "instructions/eval_plans"
TL = "train_logs/PerAct2"
TASKS = ["bimanual_sweep_to_dustpan", "bimanual_take_tray_out_of_oven"]
# Best SBRS-ladder checkpoint of each run (trained bias + 3/1 condition).
METHODS = [{"id": "nowrist4_base_s50k", "checkpoint": f"{TL}/peract2_orbital_nowrist4_base/interm_step_50000.pth"},
           {"id": "nowrist4_vid_pooled_s40k", "checkpoint": f"{TL}/peract2_orbital_nowrist4_vid_pooled/interm_step_40000.pth",
            "overrides": {"visual_num_history": 5, "eval_proprio_history_order": "past_to_current"}}]
# (source plan, kept calibrations or None for all); campaign ids stay the source's, so results sit beside the 13-task models.
SOURCES = [("idlt_task_heldout_group_v01", ["calibrated", "unknown-e10deg-t10cm"]),
           ("idlt_extreme_task_heldout_group_v01", None),
           ("idlt_scaled_extreme_task_heldout_group_v01", None)]


def restrict(plan, keep):
    plan["methods"] = METHODS
    plan["tasks"] = TASKS
    plan["task_viewpoints"] = {t: plan["task_viewpoints"][t] for t in TASKS}
    if keep:
        plan["calibrations"] = keep
    plan["task_calibrations"] = {t: [c for c in plan["task_calibrations"][t]
                                     if any(c == k or c.endswith(f"-{k}-v01") for k in plan["calibrations"])] for t in TASKS}
    return plan


def write(plan, name):
    out = PLANS / f"{name}_nowrist4.json"
    out.write_text(json.dumps(plan, indent=2) + "\n")
    cells = sum(len(v) for v in plan["task_calibrations"].values()) * len(plan["methods"])
    print(out.name, plan["campaign_id"], cells, "cells")
    return cells


total = 0
for src, keep in SOURCES:
    total += write(restrict(json.loads((PLANS / f"{src}.json").read_text()), keep), src)
plan = json.loads((PLANS / "idlt_scaled_extreme_task_heldout_group_v01.json").read_text())
plan["campaign_id"] = "idlt_scaled_extreme_task_heldout_group_v01"
plan["calibrations"] = ["unknown-e30deg-t30cm", "unknown-e40deg-t40cm"]
plan["task_calibrations"] = {t: [f"seen-{plan['task_viewpoints'][t]['spawn_camera_group'].lower()}-medium-{c}-v01"
                                 for c in plan["calibrations"]] for t in plan["tasks"]}
total += write(restrict(plan, None), "idlt_scaled_extreme_3040_task_heldout_group_v01")
print("total cells", total)
