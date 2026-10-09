#!/usr/bin/env python3
"""Fill the missing IDLT levels and G7 suites for the 200k checkpoints that only have IDLT calibrated and 60/60."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "instructions/eval_plans"
TL = "train_logs/PerAct2"
VIDEO = {"visual_num_history": 5, "eval_proprio_history_order": "past_to_current"}
RUNS = [("exp1a_adaln_percam_s200k_ema", "peract2_orbital_vid_pooled_randhist_adaln_percam_scratch", "ema"),
        ("exp1c_adaln_static_s200k_ema", "peract2_orbital_vid_pooled_randhist_adaln_static_scratch", "ema"),
        ("exp2_direct_randhist_s200k_ema", "peract2_orbital_vid_direct_randhist_scratch", "ema"),
        ("exp3_randhist_deltam_s200k_ema", "peract2_orbital_vid_pooled_randhist_deltam_scratch", "ema"),
        ("regtest_a_register_s200k", "peract2_orbital_vid_pooled_noalign_randhist_scratch", "raw")]
METHODS = [{"id": mid, "checkpoint": f"{TL}/{run}/interm_step_200000.pth", "overrides": {**VIDEO, "eval_weights": w}} for mid, run, w in RUNS]
# (source plan, calibration levels already run at 200k and therefore dropped)
SOURCES = [("idlt_task_heldout_group_v01", ["calibrated"]), ("idlt_extreme_task_heldout_group_v01", []),
           ("idlt_scaled_extreme_task_heldout_group_v01", ["unknown-e60deg-t60cm"]),
           ("g7_extreme_opposite_v01", []), ("g7_rotation_only_v01", []), ("g7_opposite_camera_sbrs_v01", [])]

total = 0
for src, drop in SOURCES:
    plan = json.loads((PLANS / f"{src}.json").read_text())
    plan["methods"] = METHODS
    plan["calibrations"] = [c for c in plan["calibrations"] if c not in drop]
    plan["task_calibrations"] = {t: [c for c in cs if not any(c == d or c.endswith(d + "-v01") for d in drop)] for t, cs in plan["task_calibrations"].items()}
    out = PLANS / f"{src}_fill200k.json"
    out.write_text(json.dumps(plan, indent=2) + "\n")
    cells = sum(len(v) for v in plan["task_calibrations"].values()) * len(METHODS)
    total += cells
    print(out.name, plan["campaign_id"], plan["calibrations"], cells, "cells")
plan = json.loads((PLANS / "idlt_scaled_extreme_task_heldout_group_v01.json").read_text())
plan.update(campaign_id="idlt_scaled_extreme_task_heldout_group_v01", methods=METHODS, calibrations=["unknown-e30deg-t30cm", "unknown-e40deg-t40cm"], task_calibrations={})
for task in plan["tasks"]:
    g = plan["task_viewpoints"][task]["spawn_camera_group"].lower()
    plan["task_calibrations"][task] = [f"seen-{g}-medium-unknown-e30deg-t30cm-v01", f"seen-{g}-medium-unknown-e40deg-t40cm-v01"]
(PLANS / "idlt_scaled_extreme_3040_task_heldout_group_v01_fill200k.json").write_text(json.dumps(plan, indent=2) + "\n")
total += 26 * len(METHODS)
print("idlt 30/40:", 26 * len(METHODS), "cells; total", total)
