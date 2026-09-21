"""Make Direct/Refine Rope3D-only plans for the existing G7 extremes."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
methods = [
    {"id": "direct_fullpatch_rope3d", "checkpoint": "train_logs/PerAct2/peract2_orbital_vid_deltam_direct_fullpatch_rope3d/best.pth",
     "overrides": {"visual_num_history": 5, "proprio_num_history": 3, "eval_proprio_history_order": "past_to_current"}},
    {"id": "refine_fullpatch_rope3d", "checkpoint": "train_logs/PerAct2/peract2_orbital_vid_deltam_refine_fullpatch_rope3d/best.pth",
     "overrides": {"visual_num_history": 5, "proprio_num_history": 3, "eval_proprio_history_order": "past_to_current"}},
]
for source, campaign in [
    ("g7_extreme_opposite_v01.json", "g7_extreme_opposite_rope3d_v01"),
    ("g7_rotation_only_v01.json", "g7_rotation_only_rope3d_v01"),
]:
    plan = json.loads((ROOT / "instructions/eval_plans" / source).read_text())
    plan["campaign_id"] = campaign
    plan["methods"] = methods
    plan["description"] += " Direct and Refine full-patch Rope3D checkpoints."
    out = ROOT / "instructions/eval_plans" / f"{campaign}.json"
    out.write_text(json.dumps(plan, indent=2) + "\n")
    print(out, len(methods) * len(plan["tasks"]) * len(next(iter(plan["task_calibrations"].values()))))
