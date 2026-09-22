"""Create the small 10-rollout G7 SBRS 200 cm comparison."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
source = json.loads((ROOT / "instructions/eval_plans/g7_extreme_opposite_v01.json").read_text())
source["campaign_id"] = "g7_sbrs_200cm_10rollouts_v01"
source["description"] = "Six G7 SBRS representative tasks, 200 cm opposing translation, 10 rollouts; Base vs VideoPooled-Refine."
source["methods"] = [
    {"id": "base_s160k", "checkpoint": "train_logs/PerAct2/peract2_orbital_new_external_only_miscal_base_200k/interm_step_160000.pth"},
    {"id": "video_pooled_refine_s084k", "checkpoint": "train_logs/PerAct2/peract2_orbital_video_deltam_external_warmstart_k5v_k3p_a5000_resume/best.pth",
     "overrides": {"visual_num_history": 5, "proprio_num_history": 3, "eval_proprio_history_order": "past_to_current"}},
]
source["calibrations"] = ["opposite-200cm"]
source["task_calibrations"] = {task: ["g7-opposite-200cm-v01"] for task in source["tasks"]}
source["runtime"]["num_demos_total"] = 10
out = ROOT / "instructions/eval_plans/g7_sbrs_200cm_10rollouts_v01.json"
out.write_text(json.dumps(source, indent=2) + "\n")
print(out, len(source["methods"]) * len(source["tasks"]), "cells")
