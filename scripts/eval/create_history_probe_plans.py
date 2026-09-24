#!/usr/bin/env python3
"""Eval-only probes of what VideoPooled's robustness depends on, at G7 opposing translation 200 cm (6 SBRS tasks x 20)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "instructions/eval_plans"
VIDEO = {"visual_num_history": 5, "eval_proprio_history_order": "past_to_current"}
VIDEO_POOLED = {"id": "video_pooled_s084k", "checkpoint": "train_logs/PerAct2/peract2_orbital_video_deltam_external_warmstart_k5v_k3p_a5000_resume/best.pth", "overrides": VIDEO}
RANDHIST = {"id": "video_pooled_randhist_notime_s050k", "checkpoint": "train_logs/PerAct2/peract2_orbital_vid_deltam_pooled_randhist_notime/interm_step_50000.pth", "overrides": VIDEO}
NO_VIEW_ALIGN = {"id": "video_pooled_no_view_align_s140k", "checkpoint": "train_logs/PerAct2/peract2_orbital_vid_deltam_pooled_no_view_align/interm_step_140000.pth", "overrides": VIDEO}
BASE = {"id": "base_s160k", "checkpoint": "train_logs/PerAct2/peract2_orbital_new_external_only_miscal_base_200k/interm_step_160000.pth"}
PROBES = {
    # All K-1 past slots hold the current frame: does history content matter at all?
    "hist_current_only": ([VIDEO_POOLED, RANDHIST, NO_VIEW_ALIGN], {"eval_history_probe": "current_only"}),
    # Past slots come from the previous episode (different scene): does it use this episode's past?
    "hist_other_episode": ([VIDEO_POOLED, RANDHIST], {"eval_history_probe": "other_episode"}),
    # Orbital cameras get black RGB and out-of-workspace points: how much rests on the wrist cameras?
    "external_blackout": ([VIDEO_POOLED, BASE], {"eval_blackout_cameras": [0, 1]}),
}

src = json.loads((PLANS / "g7_extreme_opposite_v01.json").read_text())
for name, (methods, probe) in PROBES.items():
    plan = json.loads(json.dumps(src))
    plan["campaign_id"] = f"g7_opposite200_probe_{name}_v01"
    plan["description"] = f"G7 opposing translation 200 cm, 6 SBRS tasks x 20 rollouts, eval-only probe: {name}."
    plan["methods"] = methods
    plan["calibrations"] = ["opposite-200cm"]
    plan["task_calibrations"] = {t: ["g7-opposite-200cm-v01"] for t in plan["tasks"]}
    plan["runtime"]["overrides"] = {**plan["runtime"].get("overrides", {}), **probe}
    out = PLANS / f"{plan['campaign_id']}.json"
    out.write_text(json.dumps(plan, indent=2) + "\n")
    print(out.name, len(methods) * len(plan["tasks"]), "cells")

# Current-frame-only across the full G7 extremes, for all three history models.
for src_name in ("g7_extreme_opposite_v01", "g7_rotation_only_v01"):
    plan = json.loads((PLANS / f"{src_name}.json").read_text())
    plan["campaign_id"] = f"{src_name.replace('_v01', '')}_hist_current_only_v01"
    plan["description"] += " Eval-only probe: history slots hold the current frame."
    plan["methods"] = [VIDEO_POOLED, NO_VIEW_ALIGN, RANDHIST]
    plan["runtime"]["overrides"] = {**plan["runtime"].get("overrides", {}), "eval_history_probe": "current_only"}
    out = PLANS / f"{plan['campaign_id']}.json"
    out.write_text(json.dumps(plan, indent=2) + "\n")
    print(out.name, sum(len(v) for v in plan["task_calibrations"].values()) * len(plan["methods"]), "cells")
