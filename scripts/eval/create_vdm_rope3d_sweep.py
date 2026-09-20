"""Build the calibration sweep (clean, 5/5, 10/10, 20/20, 40/40, 60/60) for the rope3d runs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTR = ROOT / "instructions"
LEVELS = ["unknown-e5deg-t5cm", "unknown-e10deg-t10cm", "unknown-e20deg-t20cm",
          "unknown-e40deg-t40cm", "unknown-e60deg-t60cm"]

# 5/10/20 live in miscal_base, 40/60 in scaled_extreme; same composition and base
# file, and byte-identical "calibrated", so merging is safe.
base = json.loads((INSTR / "eval_calibrations_miscal_base_v01.json").read_text())
extreme = json.loads((INSTR / "eval_calibrations_idlt_scaled_extreme_v01.json").read_text())
assert base["composition"] == extreme["composition"]
assert base["realizations"]["calibrated"] == extreme["realizations"]["calibrated"]

keep = {"calibrated": base["realizations"]["calibrated"]}
for src in (base, extreme):
    for name, value in src["realizations"].items():
        if any(f"-{lvl}-" in f"-{name}-" for lvl in LEVELS):
            keep[name] = value
registry = dict(base)
registry["realizations"] = keep
registry["source"] = {"merged_from": ["eval_calibrations_miscal_base_v01.json",
                                      "eval_calibrations_idlt_scaled_extreme_v01.json"],
                      "base": base.get("source")}
reg_path = INSTR / "eval_calibrations_vdm_rope3d_sweep_v01.json"
reg_path.write_text(json.dumps(registry, indent=2) + "\n")

plan = json.loads((INSTR / "eval_plans/idlt_task_heldout_group_v01.json").read_text())
plan["campaign_id"] = "vdm_rope3d_sweep_task_heldout_group_v01"
plan["description"] = ("Vid-DeltaM full-patch + patch-RoPE, Direct vs Refine, calibration sweep "
                       "clean through 60/60, task_heldout_group.")
plan["calibration_registry"] = "instructions/eval_calibrations_vdm_rope3d_sweep_v01.json"
plan["output_root"] = "/grogu/datasets/hbhatia/3dfa_online_eval_100rollouts"
plan["methods"] = [
    {"id": "vdm-direct-fullpatch-rope3d-best",
     "checkpoint": "train_logs/PerAct2/peract2_orbital_vid_deltam_direct_fullpatch_rope3d/best.pth"},
    {"id": "vdm-refine-fullpatch-rope3d-best",
     "checkpoint": "train_logs/PerAct2/peract2_orbital_vid_deltam_refine_fullpatch_rope3d/best.pth"},
]
plan["calibrations"] = ["calibrated"] + LEVELS
plan["task_calibrations"] = {}
for task in plan["tasks"]:
    g = plan["task_viewpoints"][task]["spawn_camera_group"].lower()
    plan["task_calibrations"][task] = ["calibrated"] + [f"seen-{g}-medium-{lvl}-v01" for lvl in LEVELS]
plan["runtime"]["num_demos_total"] = 100
plan_path = INSTR / "eval_plans/vdm_rope3d_sweep_task_heldout_group_v01.json"
plan_path.write_text(json.dumps(plan, indent=2) + "\n")

cells = len(plan["methods"]) * sum(len(v) for v in plan["task_calibrations"].values())
print(f"registry: {len(keep)} realizations -> {reg_path.name}")
print(f"plan: {len(plan['methods'])} methods x {len(plan['tasks'])} tasks x 6 calibrations = {cells} cells")
