#!/usr/bin/env python3
"""Exp 8 handover evals at 50k: 8a with the planner, 8b with the IK chunk controller; held-out G4 and seen G1."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "instructions/eval_plans"
TL = "train_logs/PerAct2"
T = "bimanual_handover_item"
METHODS = [
    {"id": "exp8a_keypose_path_s50k_ema", "checkpoint": f"{TL}/peract2_orbital_handover_dense_base_keypose_path/interm_step_50000.pth",
     "overrides": {"eval_weights": "ema"}},
    {"id": "exp8b_chunk8_s50k_ema_ik", "checkpoint": f"{TL}/peract2_orbital_handover_dense_base_chunk8/interm_step_50000.pth",
     "overrides": {"eval_weights": "ema", "eval_controller": "ik", "max_steps": 60}},
]
SEEN_G1 = {"id": "g1_task_seen_group", "spawn_camera_group": "G1", "regime": "task_seen_group", "cameras_file": "instructions/orbital_cameras_grouped.json"}


EXEC4 = [{"id": "exp8b_chunk8_s50k_ema_ik_exec4", "checkpoint": METHODS[1]["checkpoint"],
          "overrides": {"eval_weights": "ema", "eval_controller": "ik", "max_steps": 120,
                        "eval_execute_steps": 4, "eval_proprio_stride": 8}}]


def plan(src, name, calibration, viewpoint=None, demos=100, video=False, methods=METHODS):
    d = json.loads((PLANS / f"{src}.json").read_text())
    d["campaign_id"] = f"exp8_{name}_v01"
    d["description"] = f"Exp 8 handover: {name}, {demos} rollouts; 8a planner, 8b IK chunk controller."
    d["tasks"] = [T]
    d["task_viewpoints"] = {T: viewpoint or d["task_viewpoints"][T]}
    d["task_calibrations"] = {T: [calibration]}
    d["calibrations"] = [calibration.split("medium-")[-1].replace("-v01", "") if calibration != "calibrated" else "calibrated"]
    d["methods"] = [dict(m, overrides=dict(m["overrides"], save_video=True)) for m in methods] if video else methods
    d["runtime"]["num_demos_total"] = demos
    out = PLANS / f"exp8_{name}_v01.json"
    out.write_text(json.dumps(d, indent=2) + "\n")
    print(out.name, len(methods), "cells")


plan("idlt_task_heldout_group_v01", "heldout_g4_calibrated", "calibrated")
plan("idlt_scaled_extreme_task_heldout_group_v01", "heldout_g4_e60t60", "seen-g4-medium-unknown-e60deg-t60cm-v01")
plan("idlt_task_heldout_group_v01", "seen_g1_calibrated", "calibrated", viewpoint=SEEN_G1)
plan("idlt_task_heldout_group_v01", "seen_g1_calibrated_videos", "calibrated", viewpoint=SEEN_G1, demos=5, video=True)
# 8b re-planned every 4 steps: run the first 4 of each 8-step chunk, proprio still at the trained 8-step stride.
plan("idlt_task_heldout_group_v01", "heldout_g4_calibrated_exec4", "calibrated", methods=EXEC4)
plan("idlt_scaled_extreme_task_heldout_group_v01", "heldout_g4_e60t60_exec4", "seen-g4-medium-unknown-e60deg-t60cm-v01", methods=EXEC4)
plan("idlt_task_heldout_group_v01", "seen_g1_calibrated_exec4", "calibrated", viewpoint=SEEN_G1, methods=EXEC4)
plan("idlt_task_heldout_group_v01", "seen_g1_calibrated_videos_exec4", "calibrated", viewpoint=SEEN_G1, demos=5, video=True, methods=EXEC4)

# Exp 8c/8d: 8a/8b retrained with the trajectory step-embedding fix.
FIX = [{"id": "exp8c_keypose_path_stepfix_s50k_ema", "checkpoint": f"{TL}/peract2_orbital_handover_dense_base_keypose_path_stepfix/interm_step_50000.pth",
        "overrides": {"eval_weights": "ema"}},
       {"id": "exp8d_chunk8_stepfix_s50k_ema_ik", "checkpoint": f"{TL}/peract2_orbital_handover_dense_base_chunk8_stepfix/interm_step_50000.pth",
        "overrides": {"eval_weights": "ema", "eval_controller": "ik", "max_steps": 60}},
       {"id": "exp8d_chunk8_stepfix_s50k_ema_ik_exec4", "checkpoint": f"{TL}/peract2_orbital_handover_dense_base_chunk8_stepfix/interm_step_50000.pth",
        "overrides": {"eval_weights": "ema", "eval_controller": "ik", "max_steps": 120, "eval_execute_steps": 4, "eval_proprio_stride": 8}}]
plan("idlt_task_heldout_group_v01", "stepfix_heldout_g4_calibrated", "calibrated", methods=FIX)
plan("idlt_scaled_extreme_task_heldout_group_v01", "stepfix_heldout_g4_e60t60", "seen-g4-medium-unknown-e60deg-t60cm-v01", methods=FIX)
plan("idlt_task_heldout_group_v01", "stepfix_seen_g1_calibrated", "calibrated", viewpoint=SEEN_G1, methods=FIX)
plan("idlt_task_heldout_group_v01", "stepfix_seen_g1_calibrated_videos", "calibrated", viewpoint=SEEN_G1, demos=5, video=True, methods=FIX)
