"""Create the 200 cm Base/VideoPooled run with per-demo GIF rollouts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
source = json.loads((ROOT / "instructions/eval_plans/g7_sbrs_200cm_10rollouts_v01.json").read_text())
source["campaign_id"] = "g7_sbrs_200cm_10rollouts_videos_v01"
source["description"] += " Saving concatenated four-camera GIF rollouts."
source["runtime"].setdefault("overrides", {})["save_video"] = True
out = ROOT / "instructions/eval_plans/g7_sbrs_200cm_10rollouts_videos_v01.json"
out.write_text(json.dumps(source, indent=2) + "\n")
print(out, 12, "cells")
