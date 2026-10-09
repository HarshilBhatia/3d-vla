#!/usr/bin/env python3
"""IDLT task-heldout and G7 extreme protocols on the 200k EMA checkpoints of exp 4a (pre-LN extractor) and 4b (input skip)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "instructions/eval_plans"
TL = "train_logs/PerAct2"
VIDEO = {"visual_num_history": 5, "eval_proprio_history_order": "past_to_current", "eval_weights": "ema"}
GROUPS = {"exp4a": [("exp4a_prenorm_s200k_ema", "peract2_orbital_vid_pooled_randhist_deltam_prenorm_scratch", 200000)],
          "exp4b": [("exp4b_inputskip_s200k_ema", "peract2_orbital_vid_pooled_randhist_deltam_inputskip_scratch", 200000)]}
# Campaign ids stay those of the earlier runs, so the new method directories sit beside A, C and VideoPooled.
SOURCES = ["idlt_task_heldout_group_v01", "idlt_extreme_task_heldout_group_v01", "idlt_scaled_extreme_task_heldout_group_v01",
           "g7_extreme_opposite_v01", "g7_rotation_only_v01", "g7_opposite_camera_sbrs_v01"]


def write(plan, name, suffix):
    out = PLANS / f"{name}_{suffix}.json"
    out.write_text(json.dumps(plan, indent=2) + "\n")
    cells = sum(len(v) for v in plan["task_calibrations"].values()) * len(plan["methods"])
    print(out.name, plan["campaign_id"], cells, "cells")
    return cells


for suffix, runs in GROUPS.items():
    methods = [{"id": mid, "checkpoint": f"{TL}/{run}/interm_step_{step}.pth", "overrides": VIDEO} for mid, run, step in runs]
    total = 0
    for src in SOURCES:
        plan = json.loads((PLANS / f"{src}.json").read_text())
        plan["methods"] = methods
        total += write(plan, src, suffix)
    plan = json.loads((PLANS / "idlt_scaled_extreme_task_heldout_group_v01.json").read_text())
    plan["campaign_id"] = "idlt_scaled_extreme_task_heldout_group_v01"
    plan["methods"] = methods
    plan["calibrations"] = ["unknown-e30deg-t30cm", "unknown-e40deg-t40cm"]
    plan["task_calibrations"] = {}
    for task in plan["tasks"]:
        g = plan["task_viewpoints"][task]["spawn_camera_group"].lower()
        plan["task_calibrations"][task] = [f"seen-{g}-medium-unknown-e30deg-t30cm-v01", f"seen-{g}-medium-unknown-e40deg-t40cm-v01"]
    total += write(plan, "idlt_scaled_extreme_3040_task_heldout_group_v01", suffix)
    print(suffix, "total cells", total)
