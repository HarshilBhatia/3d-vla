"""Create a G7 rotation-only calibration sweep (no translation)."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
angles = (20, 90, 180, 270)
cameras = ["orbital_left", "orbital_right", "wrist_left", "wrist_right"]
I = np.eye(4)
realizations = {"g7-rot-clean-v01": {"description": "Identity calibration.", "transforms": {c: I.tolist() for c in cameras}}}
for deg in angles:
    a = np.deg2rad(deg)
    R = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    transforms = {c: I.tolist() for c in cameras}
    left = I.copy(); right = I.copy()
    left[:3, :3] = R; right[:3, :3] = R
    transforms["orbital_left"], transforms["orbital_right"] = left.tolist(), right.tolist()
    realizations[f"g7-rot-{deg}deg-v01"] = {
        "description": f"G7 orbital cameras rotated {deg} degrees about world Z; zero translation.",
        "transforms": transforms,
    }
registry = {
    "schema_version": 1, "camera_order": cameras,
    "composition": "T_observed = T_true with direct world-Z orbital-camera rotations",
    "source": {"camera_group": "G7", "axis": "world +Z", "translation": "zero"},
    "realizations": realizations,
}
(ROOT / "instructions/eval_calibrations_g7_rotation_only_v01.json").write_text(json.dumps(registry, indent=2) + "\n")

tasks = ["bimanual_handover_item_easy", "bimanual_lift_ball", "bimanual_push_box",
         "bimanual_pick_plate", "bimanual_pick_laptop", "bimanual_straighten_rope"]
methods = [
    {"id": "base_s160k", "checkpoint": "train_logs/PerAct2/peract2_orbital_new_external_only_miscal_base_200k/interm_step_160000.pth"},
    {"id": "deltam_s140k", "checkpoint": "train_logs/PerAct2/peract2_orbital_new_external_only_miscal_deltam_external_200k/interm_step_140000.pth"},
    {"id": "video_pooled_s084k", "checkpoint": "train_logs/PerAct2/peract2_orbital_video_deltam_external_warmstart_k5v_k3p_a5000_resume/best.pth", "overrides": {"visual_num_history": 5, "eval_proprio_history_order": "past_to_current"}},
    {"id": "video_fullpatch_best", "checkpoint": "train_logs/PerAct2/peract2_orbital_vid_deltam_fullpatch_finetune_300k_a6000_b64/best.pth", "overrides": {"visual_num_history": 5, "eval_proprio_history_order": "past_to_current"}},
]
cal_ids = [f"g7-rot-{x}deg-v01" for x in angles]
plan = {
    "schema_version": 1, "campaign_id": "g7_rotation_only_v01",
    "description": "G7 unseen camera group; six SBRS tasks; orbital-camera world-Z rotations 20/90/180/270 degrees; zero translation.",
    "calibration_registry": "instructions/eval_calibrations_g7_rotation_only_v01.json",
    "output_root": "/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts",
    "methods": methods, "calibrations": [f"rot-{x}deg" for x in angles],
    "tasks": tasks, "task_viewpoints": {}, "task_calibrations": {},
    "runtime": {"data": "orbital_peract2_nfs", "dataset": "OrbitalPeract2", "bimanual": True,
        "data_dir": "/grogu/datasets/hbhatia/peract2_test/peract2_test", "headless": True,
        "max_tries": 1, "eval_use_depth2cloud": True, "num_demos_total": 20,
        "overrides": {"scene_sampling": "fps", "eval_proprio_history_order": "past_to_current"}},
}
for task in tasks:
    plan["task_viewpoints"][task] = {"id": "g7_rotation_unknown", "spawn_camera_group": "G7", "regime": "fully_unknown_lab", "cameras_file": "instructions/orbital_cameras_grouped.json"}
    plan["task_calibrations"][task] = cal_ids
(ROOT / "instructions/eval_plans/g7_rotation_only_v01.json").write_text(json.dumps(plan, indent=2) + "\n")
print("created", len(methods) * len(tasks) * len(cal_ids), "cells")
