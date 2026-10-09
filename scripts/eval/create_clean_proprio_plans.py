#!/usr/bin/env python3
"""IDLT task-heldout calibrated + 60/60 with the eval_clean_proprio oracle (TODO 16) for 1b, A and exp 3 at 200k; --dry-run writes a 2-demo smoke plan."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "instructions/eval_plans"
TL = "train_logs/PerAct2"
BASE = {"visual_num_history": 5, "eval_proprio_history_order": "past_to_current", "eval_clean_proprio": True}
METHODS = [
    {"id": "exp1b_adaln_pooled_s200k_ema_clean_proprio", "checkpoint": f"{TL}/peract2_orbital_vid_pooled_randhist_adaln_pooled_scratch/interm_step_200000.pth",
     "overrides": {**BASE, "eval_weights": "ema"}},
    {"id": "regtest_a_register_s200k_clean_proprio", "checkpoint": f"{TL}/peract2_orbital_vid_pooled_noalign_randhist_scratch/interm_step_200000.pth",
     "overrides": {**BASE, "eval_weights": "raw"}},
    {"id": "exp3_randhist_deltam_s200k_ema_clean_proprio", "checkpoint": f"{TL}/peract2_orbital_vid_pooled_randhist_deltam_scratch/interm_step_200000.pth",
     "overrides": {**BASE, "eval_weights": "ema"}},
]
# (source plan, calibration level kept)
JOBS = [("idlt_task_heldout_group_v01", "calibrated"), ("idlt_scaled_extreme_task_heldout_group_v01", "e60deg-t60cm")]

p = argparse.ArgumentParser()
p.add_argument("--dry-run", action="store_true", help="one task, 2 demos, 1b only, scratch output root")
a = p.parse_args()
total = 0
for src, level in JOBS:
    plan = json.loads((PLANS / f"{src}.json").read_text())
    plan["methods"] = METHODS
    plan["calibrations"] = [c for c in plan["calibrations"] if c.endswith(level)]
    plan["task_calibrations"] = {t: [c for c in cs if level in c] for t, cs in plan["task_calibrations"].items()}
    suffix = "clean_proprio"
    if a.dry_run:
        task = "bimanual_push_box"
        plan["methods"] = [{**METHODS[0], "overrides": {**METHODS[0]["overrides"], "num_demos": 2}}]
        plan["task_calibrations"] = {task: plan["task_calibrations"][task]}
        plan["tasks"] = [task]
        plan["task_viewpoints"] = {task: plan["task_viewpoints"][task]}
        plan["output_root"] = "/grogu/datasets/hbhatia/3dfa_online_eval_dryrun"
        suffix = "clean_proprio_dryrun"
    out = PLANS / f"{src}_{suffix}.json"
    out.write_text(json.dumps(plan, indent=2) + "\n")
    n = sum(len(v) for v in plan["task_calibrations"].values()) * len(plan["methods"])
    total += n
    print(out.name, plan["campaign_id"], plan["calibrations"], n, "cells")
print("total", total)
