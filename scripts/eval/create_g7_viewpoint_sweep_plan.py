#!/usr/bin/env python3
"""Create the G7 viewpoint-sweep plan: calibrated extrinsics, cameras physically moved, six SBRS tasks."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAMERAS = "instructions/orbital_cameras_g7_sweep_v01.json"
VIDEO = {"visual_num_history": 5, "eval_proprio_history_order": "past_to_current"}
methods = [
    {"id": "base_s160k", "checkpoint": "train_logs/PerAct2/peract2_orbital_new_external_only_miscal_base_200k/interm_step_160000.pth"},
    {"id": "video_pooled_s084k", "checkpoint": "train_logs/PerAct2/peract2_orbital_video_deltam_external_warmstart_k5v_k3p_a5000_resume/best.pth", "overrides": VIDEO},
    {"id": "video_fullpatch_best", "checkpoint": "train_logs/PerAct2/peract2_orbital_vid_deltam_fullpatch_finetune_300k_a6000_b64/best.pth", "overrides": VIDEO},
]
groups = [g["group"] for g in json.loads((ROOT / CAMERAS).read_text())]
plan = {
    "schema_version": 1, "campaign_id": "g7_viewpoint_sweep_v01",
    "description": "G7 plus ten moved orbital-camera pairs (elevation, azimuth, radius, combined); calibrated extrinsics; six SBRS tasks.",
    "calibration_registry": "instructions/eval_calibrations_idlt_v01.json",
    "output_root": "/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts",
    "methods": methods, "calibrations": ["calibrated"],
    "tasks": ["bimanual_handover_item_easy", "bimanual_lift_ball", "bimanual_push_box",
              "bimanual_pick_plate", "bimanual_pick_laptop", "bimanual_straighten_rope"],
    "viewpoints": [{"id": g.lower(), "spawn_camera_group": g, "regime": "g7_viewpoint_sweep", "cameras_file": CAMERAS} for g in groups],
    "runtime": {"data": "orbital_peract2_nfs", "dataset": "OrbitalPeract2", "bimanual": True,
                "data_dir": "/grogu/datasets/hbhatia/peract2_test/peract2_test", "headless": True,
                "max_tries": 1, "eval_use_depth2cloud": True, "num_demos_total": 20,
                "overrides": {"scene_sampling": "fps", "eval_proprio_history_order": "past_to_current"}},
}
out = ROOT / "instructions/eval_plans/g7_viewpoint_sweep_v01.json"
out.write_text(json.dumps(plan, indent=2) + "\n")
print(out, len(methods) * len(groups) * len(plan["tasks"]), "cells")
