#!/usr/bin/env python3
"""Exp 9 evals: standard-PerAct2 bimanual_lift_ball, calibrated and front-camera miscalibration, 100 rollouts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "instructions/eval_plans"
TL = "train_logs/PerAct2"
CALS = ["calibrated", "front-e3deg-t1cm-v01", "front-e5deg-t5cm-v01", "front-e10deg-t10cm-v01", "front-e20deg-t20cm-v01"]
CHUNK = {"eval_weights": "ema", "eval_controller": "ik", "max_steps": 60, "prediction_len": 8}
SEG = {"eval_weights": "ema", "prediction_len": 50}
METHODS = [
    {"id": "exp9a_chunk8_s50k_ema_ik", "checkpoint": f"{TL}/peract2_lift_ball_dense_base_chunk8/interm_step_50000.pth", "overrides": CHUNK},
    {"id": "exp9b_chunk8_noise_s50k_ema_ik", "checkpoint": f"{TL}/peract2_lift_ball_dense_base_chunk8_noise/interm_step_50000.pth", "overrides": CHUNK},
    {"id": "exp9c_segment50_s100k_ema_all50", "checkpoint": f"{TL}/peract2_lift_ball_keypose_segment50_base/interm_step_100000.pth", "overrides": SEG},
    {"id": "exp9c_segment50_s100k_ema_last", "checkpoint": f"{TL}/peract2_lift_ball_keypose_segment50_base/interm_step_100000.pth",
     "overrides": dict(SEG, eval_execute_steps=-1)},
    {"id": "exp9d_segment50_noise_s100k_ema_all50", "checkpoint": f"{TL}/peract2_lift_ball_keypose_segment50_base_noise/interm_step_100000.pth", "overrides": SEG},
    {"id": "exp9d_segment50_noise_s100k_ema_last", "checkpoint": f"{TL}/peract2_lift_ball_keypose_segment50_base_noise/interm_step_100000.pth",
     "overrides": dict(SEG, eval_execute_steps=-1)},
]
base = json.loads((PLANS / "peract2_original_clean_v01.json").read_text())
for name, demos, video in [("exp9_lift_ball_v01", 100, False), ("exp9_lift_ball_videos_v01", 5, True)]:
    d = dict(base)
    d["campaign_id"] = name
    d["description"] = f"Exp 9: standard-PerAct2 lift_ball, front-camera miscalibration sweep, {demos} rollouts."
    d["calibration_registry"] = "instructions/eval_calibrations_peract2_standard_front_miscal_v01.json"
    d["output_root"] = "/grogu/datasets/hbhatia/3dfa_online_eval_100rollouts"
    d["calibrations"] = CALS if not video else ["calibrated"]
    d["tasks"] = ["bimanual_lift_ball"]
    d["runtime"] = dict(base["runtime"], num_demos_total=demos, data="full")
    d["runtime"]["overrides"] = {}
    d["methods"] = [dict(m, overrides=dict(m["overrides"], save_video=True)) for m in METHODS] if video else METHODS
    (PLANS / f"{name}.json").write_text(json.dumps(d, indent=2) + "\n")
    print(name, len(d["methods"]) * len(d["calibrations"]), "cells")
