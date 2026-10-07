#!/usr/bin/env python3
"""IDLT calibrated and 60/60 (100 rollouts) for control A's 200k checkpoint, raw weights (A was trained without EMA)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "instructions/eval_plans"
METHODS = [{"id": "regtest_a_register_s200k",
            "checkpoint": "train_logs/PerAct2/peract2_orbital_vid_pooled_noalign_randhist_scratch/interm_step_200000.pth",
            "overrides": {"visual_num_history": 5, "eval_proprio_history_order": "past_to_current", "eval_weights": "raw"}}]
KEEP = [("idlt_task_heldout_group_v01", "calibrated"), ("idlt_scaled_extreme_task_heldout_group_v01", "e60deg-t60cm")]

total = 0
for src, level in KEEP:
    plan = json.loads((PLANS / f"{src}.json").read_text())
    plan["methods"] = METHODS
    plan["calibrations"] = [c for c in plan["calibrations"] if c.endswith(level)]
    plan["task_calibrations"] = {t: [c for c in cs if level in c] for t, cs in plan["task_calibrations"].items()}
    out = PLANS / f"{src}_regtest_a_s200k.json"
    out.write_text(json.dumps(plan, indent=2) + "\n")
    cells = sum(len(v) for v in plan["task_calibrations"].values()) * len(METHODS)
    total += cells
    print(out.name, plan["campaign_id"], plan["calibrations"], cells, "cells")
print("total cells", total)
