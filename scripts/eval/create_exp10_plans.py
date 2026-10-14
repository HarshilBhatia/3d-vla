#!/usr/bin/env python3
"""Exp 10 evals (8d setting variants) at 15k: handover seen G1 calibrated, held-out G4 calibrated to 60/60, videos."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "instructions/eval_plans"
TL = "train_logs/PerAct2"
T = "bimanual_handover_item"
RUNS = {"10a": ("deltam", 1), "10b": ("fullpatch_refine", 5), "10c": ("fullpatch_refine_x0", 5),
        "10d": ("fullpatch_refine_rel", 5), "10e": ("rel", 1), "10f": ("pooled_refine", 5)}
SEEN_G1 = {"id": "g1_task_seen_group", "spawn_camera_group": "G1", "regime": "task_seen_group", "cameras_file": "instructions/orbital_cameras_grouped.json"}


def methods(video=False, step=15000, runs=None):
    out = []
    for key, (cfg, k) in RUNS.items():
        if runs is not None and key not in runs:
            continue
        o = {"eval_weights": "ema", "eval_controller": "ik", "max_steps": 60, "prediction_len": 8,
             "visual_num_history": k, "eval_proprio_history_order": "past_to_current"}
        if video:
            o["save_video"] = True
        out.append({"id": f"exp{key}_{cfg}_s{step // 1000}k_ema", "checkpoint": f"{TL}/peract2_orbital_handover_dense_chunk8_{cfg}/interm_step_{step}.pth", "overrides": o})
    return out


def plan(src, name, cal_ids, viewpoint=None, demos=25, video=False, step=15000, runs=None, version="v01"):
    d = json.loads((PLANS / f"{src}.json").read_text())
    d["campaign_id"] = f"exp10_{name}_{version}"
    d["description"] = f"Exp 10 (8d setting variants) handover: {name}, {demos} rollouts, IK chunk controller."
    d["tasks"] = [T]
    d["task_viewpoints"] = {T: viewpoint or d["task_viewpoints"][T]}
    d["task_calibrations"] = {T: cal_ids}
    d["calibrations"] = ["calibrated" if c == "calibrated" else c.split("medium-")[-1].replace("-v01", "") for c in cal_ids]
    d["methods"] = methods(video, step, runs)
    d["runtime"]["num_demos_total"] = demos
    (PLANS / f"exp10_{name}_{version}.json").write_text(json.dumps(d, indent=2) + "\n")
    print(f"exp10_{name}_{version}.json", len(cal_ids) * len(d["methods"]), "cells")


g4 = lambda *ls: [f"seen-g4-medium-unknown-e{l}deg-t{l}cm-v01" for l in ls]
# v01 at 15k (undertrained); v02 at 50k for all six (final for 10a/10e); v03 at 100k for the history runs.
for version, step, runs in [("v01", 15000, None), ("v02", 50000, None), ("v03", 100000, ["10b", "10c", "10d", "10f"])]:
    kw = dict(step=step, runs=runs, version=version)
    plan("idlt_task_heldout_group_v01", "seen_g1_calibrated", ["calibrated"], viewpoint=SEEN_G1, **kw)
    plan("idlt_task_heldout_group_v01", "heldout_g4_cal_10", ["calibrated"] + g4(10), **kw)
    plan("idlt_extreme_task_heldout_group_v01", "heldout_g4_15_20", g4(15, 20), **kw)
    plan("idlt_scaled_extreme_task_heldout_group_v01", "heldout_g4_30_60", g4(30, 40, 50, 60), **kw)
    plan("idlt_task_heldout_group_v01", "seen_g1_videos", ["calibrated"], viewpoint=SEEN_G1, demos=5, video=True, **kw)
