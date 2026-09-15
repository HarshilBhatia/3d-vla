#!/usr/bin/env python3
"""Materialize the two In-Distribution Lab Transfer (IDLT) evaluation plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS = (
    "bimanual_push_box", "bimanual_lift_ball", "bimanual_dual_push_buttons",
    "bimanual_pick_plate", "bimanual_put_item_in_drawer", "bimanual_put_bottle_in_fridge",
    "bimanual_handover_item", "bimanual_pick_laptop", "bimanual_straighten_rope",
    "bimanual_sweep_to_dustpan", "bimanual_lift_tray", "bimanual_handover_item_easy",
    "bimanual_take_tray_out_of_oven",
)
CONDITIONS = ("calibrated", "known", "supported-e3deg-t1cm", "unknown-e5deg-t5cm", "unknown-e10deg-t10cm")
METHODS = [
    {"id": "base_s160k", "checkpoint": "train_logs/PerAct2/peract2_orbital_new_external_only_miscal_base_200k/interm_step_160000.pth"},
    {"id": "deltam_s140k", "checkpoint": "train_logs/PerAct2/peract2_orbital_new_external_only_miscal_deltam_external_200k/interm_step_140000.pth"},
    {
        "id": "video_deltam_best_s084k",
        "checkpoint": "train_logs/PerAct2/peract2_orbital_video_deltam_external_warmstart_k5v_k3p_a5000_resume/best.pth",
        "overrides": {"image_space_sampling": False, "eval_proprio_history_order": "past_to_current"},
    },
]
RUNTIME = {
    "data": "orbital_peract2_nfs", "dataset": "OrbitalPeract2", "bimanual": True,
    "data_dir": "/grogu/datasets/hbhatia/peract2_test/peract2_test", "headless": True,
    "max_tries": 1, "eval_use_depth2cloud": True, "num_demos_total": 100,
}


def load_mapping(path: Path) -> dict:
    with path.open() as handle:
        mapping = json.load(handle)["tasks"]
    missing = set(TASKS) - set(mapping)
    if missing:
        raise ValueError(f"Task mapping missing: {sorted(missing)}")
    return mapping


def calibration_ids(group: str) -> list[str]:
    group = group.lower()
    return [
        "calibrated",
        f"seen-{group}-medium-known-v01",
        f"seen-{group}-medium-supported-e3deg-t1cm-v01",
        f"seen-{group}-medium-unknown-e5deg-t5cm-v01",
        f"seen-{group}-medium-unknown-e10deg-t10cm-v01",
    ]


def build_plan(regime: str, mapping: dict, registry: str, output_root: str) -> dict:
    if regime not in {"task_seen_group", "task_heldout_group"}:
        raise ValueError(regime)
    groups = {
        task: mapping[task]["train_groups"][0] if regime == "task_seen_group" else mapping[task]["eval_group"]
        for task in TASKS
    }
    task_viewpoints = {
        task: {
            "id": f"{groups[task].lower()}_{regime}",
            "spawn_camera_group": groups[task],
            "regime": regime,
            "cameras_file": "instructions/orbital_cameras_grouped.json",
        }
        for task in TASKS
    }
    return {
        "schema_version": 1,
        "campaign_id": f"idlt_{regime}_v01",
        "description": (
            "In-Distribution Lab Transfer: external-camera-only calibration robustness. "
            + ("The target task trained on its selected camera group." if regime == "task_seen_group" else "The selected camera group was seen in training for other tasks, but not this target task.")
        ),
        "calibration_registry": registry,
        "output_root": output_root,
        "methods": METHODS,
        "task_viewpoints": task_viewpoints,
        "task_calibrations": {task: calibration_ids(group) for task, group in groups.items()},
        "calibrations": list(CONDITIONS),
        "tasks": list(TASKS),
        "runtime": RUNTIME,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=REPO_ROOT / "instructions/peract2_orbital_task_group_mapping.json")
    parser.add_argument("--registry", default="instructions/eval_calibrations_idlt_v01.json")
    parser.add_argument("--output-root", default="/grogu/datasets/hbhatia/3dfa_online_eval_100rollouts")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "instructions/eval_plans")
    args = parser.parse_args()
    mapping = load_mapping(args.mapping)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for regime in ("task_seen_group", "task_heldout_group"):
        plan = build_plan(regime, mapping, args.registry, args.output_root)
        path = args.output_dir / f"{plan['campaign_id']}.json"
        path.write_text(json.dumps(plan, indent=2) + "\n")
        print(f"Wrote {path} ({len(METHODS) * len(TASKS) * len(CONDITIONS)} cells)")


if __name__ == "__main__":
    main()
