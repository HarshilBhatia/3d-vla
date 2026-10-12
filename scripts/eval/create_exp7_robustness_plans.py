#!/usr/bin/env python3
"""IDLT task-heldout and G7 protocols on the 100k EMA checkpoints of exp 7a-7f, restricted to their 4 training tasks."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "instructions/eval_plans"
TL = "train_logs/PerAct2"
TASKS = ["bimanual_handover_item", "bimanual_take_tray_out_of_oven", "bimanual_dual_push_buttons", "bimanual_pick_laptop"]
OVR = lambda k: {"visual_num_history": k, "eval_proprio_history_order": "past_to_current", "eval_weights": "ema"}
GROUPS = {"exp7a": ("exp7a_4task_vel_s100k_ema", "peract2_orbital_4task_fullpatch_direct_vel", 5),
          "exp7b": ("exp7b_4task_x0_s100k_ema", "peract2_orbital_4task_fullpatch_direct_x0", 5),
          "exp7c": ("exp7c_4task_rel_s100k_ema", "peract2_orbital_4task_fullpatch_direct_rel", 5),
          "exp7d": ("exp7d_4task_base_rel_s100k_ema", "peract2_orbital_4task_base_rel", 1),
          "exp7e": ("exp7e_4task_base_abs_s100k_ema", "peract2_orbital_4task_base_abs", 1),
          "exp7f": ("exp7f_4task_pooled_direct_s100k_ema", "peract2_orbital_4task_pooled_direct", 5)}
SOURCES = ["idlt_task_heldout_group_v01", "idlt_extreme_task_heldout_group_v01", "idlt_scaled_extreme_task_heldout_group_v01",
           "g7_extreme_opposite_v01", "g7_rotation_only_v01", "g7_opposite_camera_sbrs_v01"]


def restrict(plan):
    plan["tasks"] = [t for t in plan["tasks"] if t in TASKS]
    for key in ("task_viewpoints", "task_calibrations"):
        plan[key] = {t: v for t, v in plan[key].items() if t in TASKS}
    return plan


def write(plan, name, suffix):
    out = PLANS / f"{name}_{suffix}.json"
    out.write_text(json.dumps(plan, indent=2) + "\n")
    cells = sum(len(v) for v in plan["task_calibrations"].values()) * len(plan["methods"])
    print(out.name, plan["campaign_id"], cells, "cells")
    return cells


for suffix, (mid, run, k) in GROUPS.items():
    methods = [{"id": mid, "checkpoint": f"{TL}/{run}/interm_step_100000.pth", "overrides": OVR(k)}]
    total = 0
    for src in SOURCES:
        plan = restrict(json.loads((PLANS / f"{src}.json").read_text()))
        if plan["tasks"]:
            plan["methods"] = methods
            total += write(plan, src, suffix)
    plan = restrict(json.loads((PLANS / "idlt_scaled_extreme_task_heldout_group_v01.json").read_text()))
    plan["methods"] = methods
    plan["calibrations"] = ["unknown-e30deg-t30cm", "unknown-e40deg-t40cm"]
    plan["task_calibrations"] = {}
    for task in plan["tasks"]:
        g = plan["task_viewpoints"][task]["spawn_camera_group"].lower()
        plan["task_calibrations"][task] = [f"seen-{g}-medium-unknown-e30deg-t30cm-v01", f"seen-{g}-medium-unknown-e40deg-t40cm-v01"]
    total += write(plan, "idlt_scaled_extreme_3040_task_heldout_group_v01", suffix)
    print(suffix, "total cells", total)
