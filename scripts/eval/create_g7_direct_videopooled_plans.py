"""Create Direct VideoPooled-only plans for G7 extreme translation/rotation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
method = [{
    "id": "video_pooled_direct",
    "checkpoint": "train_logs/PerAct2/peract2_orbital_vid_deltam_direct/best.pth",
    "overrides": {"visual_num_history": 5, "proprio_num_history": 3,
                   "eval_proprio_history_order": "past_to_current"},
}]
for source, campaign in [
    ("g7_extreme_opposite_v01.json", "g7_extreme_opposite_videopooled_direct_v01"),
    ("g7_rotation_only_v01.json", "g7_rotation_only_videopooled_direct_v01"),
]:
    plan = json.loads((ROOT / "instructions/eval_plans" / source).read_text())
    plan["campaign_id"] = campaign
    plan["methods"] = method
    plan["description"] += " Direct VideoPooled checkpoint."
    out = ROOT / "instructions/eval_plans" / f"{campaign}.json"
    out.write_text(json.dumps(plan, indent=2) + "\n")
    n = len(plan["tasks"]) * len(next(iter(plan["task_calibrations"].values())))
    print(out, n, "cells")
