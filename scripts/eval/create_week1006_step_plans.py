#!/usr/bin/env python3
"""IDLT calibrated and 60/60 across exp 3's checkpoints, plus the 200k checkpoints of 1a, 1c and exp 2, to separate step from method."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "instructions/eval_plans"
TL = "train_logs/PerAct2"
VIDEO = {"visual_num_history": 5, "eval_proprio_history_order": "past_to_current", "eval_weights": "ema"}
RUNS = {"exp3_randhist_deltam": "peract2_orbital_vid_pooled_randhist_deltam_scratch",
        "exp1a_adaln_percam": "peract2_orbital_vid_pooled_randhist_adaln_percam_scratch",
        "exp1c_adaln_static": "peract2_orbital_vid_pooled_randhist_adaln_static_scratch",
        "exp2_direct_randhist": "peract2_orbital_vid_direct_randhist_scratch"}
STEPS = [("exp3_randhist_deltam", s) for s in (50000, 80000, 140000, 170000, 200000)] + \
        [(r, 200000) for r in ("exp1a_adaln_percam", "exp1c_adaln_static", "exp2_direct_randhist")]
METHODS = [{"id": f"{r}_s{s // 1000}k_ema", "checkpoint": f"{TL}/{RUNS[r]}/interm_step_{s}.pth", "overrides": VIDEO} for r, s in STEPS]
# (source plan, calibration kept): the same campaigns as A, C and this week's best-checkpoint evals.
KEEP = [("idlt_task_heldout_group_v01", "calibrated"), ("idlt_scaled_extreme_task_heldout_group_v01", "e60deg-t60cm")]

total = 0
for src, level in KEEP:
    plan = json.loads((PLANS / f"{src}.json").read_text())
    plan["methods"] = METHODS
    plan["calibrations"] = [c for c in plan["calibrations"] if c.endswith(level)]
    plan["task_calibrations"] = {t: [c for c in cs if level in c] for t, cs in plan["task_calibrations"].items()}
    out = PLANS / f"{src}_wk1006_steps.json"
    out.write_text(json.dumps(plan, indent=2) + "\n")
    cells = sum(len(v) for v in plan["task_calibrations"].values()) * len(METHODS)
    total += cells
    print(out.name, plan["campaign_id"], plan["calibrations"], cells, "cells")
print("total cells", total)
