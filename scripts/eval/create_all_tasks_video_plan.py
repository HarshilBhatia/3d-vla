#!/usr/bin/env python3
"""Record five calibrated, task-seen rollouts per task for all 13 tasks, with four-camera videos, for no-wrist task selection."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
src = json.loads((ROOT / "instructions/eval_plans/idlt_task_seen_group_v01.json").read_text())
plan = {
    "schema_version": 1, "campaign_id": "all_tasks_rollout_videos_v01",
    "description": "All 13 tasks, task-seen camera group, calibrated, 5 rollouts each with four-camera videos; Base 160k and VideoPooled 84k.",
    "calibration_registry": "instructions/eval_calibrations_idlt_v01.json",
    "output_root": "/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts",
    "methods": [
        {"id": "base_s160k", "checkpoint": "train_logs/PerAct2/peract2_orbital_new_external_only_miscal_base_200k/interm_step_160000.pth"},
        {"id": "video_pooled_s084k", "checkpoint": "train_logs/PerAct2/peract2_orbital_video_deltam_external_warmstart_k5v_k3p_a5000_resume/best.pth",
         "overrides": {"visual_num_history": 5, "eval_proprio_history_order": "past_to_current"}},
    ],
    "calibrations": ["calibrated"], "tasks": src["tasks"], "task_viewpoints": src["task_viewpoints"],
    "runtime": {"data": "orbital_peract2_nfs", "dataset": "OrbitalPeract2", "bimanual": True,
                "data_dir": "/grogu/datasets/hbhatia/peract2_test/peract2_test", "headless": True,
                "max_tries": 1, "eval_use_depth2cloud": True, "num_demos_total": 5,
                "overrides": {"scene_sampling": "fps", "eval_proprio_history_order": "past_to_current", "save_video": True}},
}
out = ROOT / "instructions/eval_plans/all_tasks_rollout_videos_v01.json"
out.write_text(json.dumps(plan, indent=2) + "\n")
print(out, len(plan["methods"]) * len(plan["tasks"]), "cells")
