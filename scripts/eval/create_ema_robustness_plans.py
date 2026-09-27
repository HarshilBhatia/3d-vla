#!/usr/bin/env python3
"""Rerun the IDLT task-heldout and G7 extreme protocols on each EMA fine-tune's best EMA checkpoint (by SBRS)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "instructions/eval_plans"
TL = "train_logs/PerAct2"
VIDEO = {"visual_num_history": 5, "eval_proprio_history_order": "past_to_current", "eval_weights": "ema"}
# (run, best EMA step by SBRS, later step on ties, video model?)
BEST = [("vidpooled", 15000, True), ("randhist", 15000, True), ("noviewalign", 15000, True),
        ("regtest_a", 20000, True), ("regtest_c", 10000, True), ("base", 20000, False)]
METHODS = [{"id": f"ema20k_{run}_s{step // 1000}k_ema",
            "checkpoint": f"{TL}/peract2_orbital_ema20k_{run}/interm_step_{step}.pth",
            "overrides": VIDEO if video else {"eval_weights": "ema"}} for run, step, video in BEST]
# Campaign ids stay those of last week's runs, so the new method directories sit beside the old ones.
SOURCES = ["idlt_task_heldout_group_v01", "idlt_extreme_task_heldout_group_v01", "idlt_scaled_extreme_task_heldout_group_v01",
           "g7_extreme_opposite_v01", "g7_rotation_only_v01", "g7_opposite_camera_sbrs_v01"]


def write(plan, name):
    out = PLANS / f"{name}_ema20k.json"
    out.write_text(json.dumps(plan, indent=2) + "\n")
    cells = sum(len(v) for v in plan["task_calibrations"].values()) * len(plan["methods"])
    print(out.name, plan["campaign_id"], cells, "cells")
    return cells


total = 0
for src in SOURCES:
    plan = json.loads((PLANS / f"{src}.json").read_text())
    plan["methods"] = METHODS
    total += write(plan, src)
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
