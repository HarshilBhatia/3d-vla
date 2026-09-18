import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for source, setting in [("idlt_task_seen_group_v01.json", "task_seen_group"), ("idlt_task_heldout_group_v01.json", "task_heldout_group")]:
    p = json.loads((ROOT / "instructions/eval_plans" / source).read_text())
    p["campaign_id"] = f"miscal_only_base_{setting}_v01"
    p["description"] = f"Miscalibration-only Base sweep, {setting}, clean through 20/20."
    p["calibration_registry"] = "instructions/eval_calibrations_miscal_base_v01.json"
    p["methods"] = [{"id": "miscal_only_base_best", "checkpoint": "train_logs/PerAct2/peract2_orbital_miscal_only_base/best.pth"}]
    p["calibrations"] = ["calibrated", "unknown-e5deg-t5cm", "unknown-e10deg-t10cm", "unknown-e15deg-t15cm", "unknown-e20deg-t20cm"]
    p["task_calibrations"] = {}
    for task in p["tasks"]:
        g = p["task_viewpoints"][task]["spawn_camera_group"].lower()
        p["task_calibrations"][task] = ["calibrated"] + [f"seen-{g}-medium-{x}-v01" for x in ["unknown-e5deg-t5cm", "unknown-e10deg-t10cm", "unknown-e15deg-t15cm", "unknown-e20deg-t20cm"]]
    (ROOT / "instructions/eval_plans" / f"miscal_only_base_{setting}_v01.json").write_text(json.dumps(p, indent=2) + "\n")
