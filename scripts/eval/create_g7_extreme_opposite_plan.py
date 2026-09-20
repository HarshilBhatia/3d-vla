"""Create the G7 100/200/500 cm opposing-camera evaluation plan."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
levels = (100, 200, 500)
camera_names = ["orbital_left", "orbital_right", "wrist_left", "wrist_right"]
groups = json.loads((ROOT / "instructions/orbital_cameras_grouped.json").read_text())
g7 = next(x for x in groups if x["group"] == "G7")
d = np.asarray(g7["left"]["pos"], float) - np.asarray(g7["right"]["pos"], float)
d /= np.linalg.norm(d)
I = np.eye(4)
realizations = {"g7-clean-v01": {"description": "Identity calibration.",
    "transforms": {c: I.tolist() for c in camera_names}}}
for cm in levels:
    left, right = I.copy(), I.copy()
    left[:3, 3] = (d * cm / 100).tolist()
    right[:3, 3] = (-d * cm / 100).tolist()
    transforms = {c: I.tolist() for c in camera_names}
    transforms["orbital_left"], transforms["orbital_right"] = left.tolist(), right.tolist()
    realizations[f"g7-opposite-{cm}cm-v01"] = {
        "description": f"G7 external cameras translated {cm}cm away from each other; no rotation.",
        "transforms": transforms,
    }
registry = {
    "schema_version": 1, "camera_order": camera_names,
    "composition": "T_observed = T_true with direct opposing external-camera translations",
    "source": {"camera_group": "G7", "direction": "camera 0 +d, camera 1 -d", "rotation": "identity"},
    "realizations": realizations,
}
(ROOT / "instructions/eval_calibrations_g7_extreme_opposite_v01.json").write_text(json.dumps(registry, indent=2) + "\n")

tasks = ["bimanual_handover_item_easy", "bimanual_lift_ball", "bimanual_push_box",
         "bimanual_pick_plate", "bimanual_pick_laptop", "bimanual_straighten_rope"]
methods = [
    {"id": "base_s160k", "checkpoint": "train_logs/PerAct2/peract2_orbital_new_external_only_miscal_base_200k/interm_step_160000.pth"},
    {"id": "deltam_s140k", "checkpoint": "train_logs/PerAct2/peract2_orbital_new_external_only_miscal_deltam_external_200k/interm_step_140000.pth"},
    {"id": "video_pooled_s084k", "checkpoint": "train_logs/PerAct2/peract2_orbital_video_deltam_external_warmstart_k5v_k3p_a5000_resume/best.pth", "overrides": {"visual_num_history": 5, "eval_proprio_history_order": "past_to_current"}},
    {"id": "video_fullpatch_best", "checkpoint": "train_logs/PerAct2/peract2_orbital_vid_deltam_fullpatch_finetune_300k_a6000_b64/best.pth", "overrides": {"visual_num_history": 5, "eval_proprio_history_order": "past_to_current"}},
]
cal_ids = [f"g7-opposite-{x}cm-v01" for x in levels]
plan = {
    "schema_version": 1, "campaign_id": "g7_extreme_opposite_v01",
    "description": "G7 unseen camera group; six SBRS tasks; 100/200/500cm opposing translations.",
    "calibration_registry": "instructions/eval_calibrations_g7_extreme_opposite_v01.json",
    "output_root": "/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts",
    "methods": methods, "calibrations": ["opposite-100cm", "opposite-200cm", "opposite-500cm"],
    "tasks": tasks, "task_viewpoints": {}, "task_calibrations": {},
    "runtime": {"data": "orbital_peract2_nfs", "dataset": "OrbitalPeract2", "bimanual": True,
                 "data_dir": "/grogu/datasets/hbhatia/peract2_test/peract2_test", "headless": True,
                 "max_tries": 1, "eval_use_depth2cloud": True, "num_demos_total": 20,
                 "overrides": {"scene_sampling": "fps", "eval_proprio_history_order": "past_to_current"}},
}
for task in tasks:
    plan["task_viewpoints"][task] = {"id": "g7_extreme_unknown", "spawn_camera_group": "G7", "regime": "fully_unknown_lab", "cameras_file": "instructions/orbital_cameras_grouped.json"}
    plan["task_calibrations"][task] = cal_ids
(ROOT / "instructions/eval_plans/g7_extreme_opposite_v01.json").write_text(json.dumps(plan, indent=2) + "\n")
print("created", len(methods) * len(tasks) * len(cal_ids), "cells")
