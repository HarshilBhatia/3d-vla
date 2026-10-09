#!/usr/bin/env python3
"""IDLT calibrated + 60/60 on task-seen groups (shortcut test) and eval-time mechanism ablations on task-heldout groups."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "instructions/eval_plans"
TL = "train_logs/PerAct2"
VIDEO = {"visual_num_history": 5, "eval_proprio_history_order": "past_to_current", "eval_weights": "ema"}
CK = {"exp3": f"{TL}/peract2_orbital_vid_pooled_randhist_deltam_scratch/interm_step_200000.pth",
      "exp4a": f"{TL}/peract2_orbital_vid_pooled_randhist_deltam_prenorm_scratch/interm_step_200000.pth",
      "exp1b": f"{TL}/peract2_orbital_vid_pooled_randhist_adaln_pooled_scratch/interm_step_200000.pth"}
SEEN = [{"id": f"{k}_s200k_ema", "checkpoint": v, "overrides": VIDEO} for k, v in CK.items()]
ABLATE = [{"id": "exp3_s200k_ema_dm_identity", "checkpoint": CK["exp3"], "overrides": {**VIDEO, "eval_ablation": "delta_m_identity"}},
          {"id": "exp4a_s200k_ema_dm_identity", "checkpoint": CK["exp4a"], "overrides": {**VIDEO, "eval_ablation": "delta_m_identity"}},
          {"id": "exp1b_s200k_ema_adaln_zero", "checkpoint": CK["exp1b"], "overrides": {**VIDEO, "eval_ablation": "camera_adaln_zero"}}]
# (source plan, level kept, methods, output suffix)
JOBS = [("idlt_task_seen_group_v01", "calibrated", SEEN, "seen"), ("idlt_scaled_extreme_task_seen_group_v01", "e60deg-t60cm", SEEN, "seen"),
        ("idlt_task_heldout_group_v01", "calibrated", ABLATE, "ablate"), ("idlt_scaled_extreme_task_heldout_group_v01", "e60deg-t60cm", ABLATE, "ablate")]
total = 0
for src, level, methods, suffix in JOBS:
    plan = json.loads((PLANS / f"{src}.json").read_text())
    plan["methods"] = methods
    plan["calibrations"] = [c for c in plan["calibrations"] if c.endswith(level)]
    plan["task_calibrations"] = {t: [c for c in cs if level in c] for t, cs in plan["task_calibrations"].items()}
    out = PLANS / f"{src}_{suffix}.json"
    out.write_text(json.dumps(plan, indent=2) + "\n")
    n = sum(len(v) for v in plan["task_calibrations"].values()) * len(methods)
    total += n
    print(out.name, plan["campaign_id"], plan["calibrations"], n, "cells")
print("total", total)
