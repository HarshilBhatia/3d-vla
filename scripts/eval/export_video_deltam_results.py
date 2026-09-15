#!/usr/bin/env python3
"""Export a visualisation-ready index of every Video-DeltaM evaluation."""

from __future__ import annotations

import json
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs" / "results"

# Keep provenance and interpretation next to the raw result roots.  The
# exporter intentionally includes diagnostics and incomplete campaigns; users
# of the JSON can filter by ``status`` / ``comparability`` rather than losing
# historical evidence.
CAMPAIGNS = [
    {
        "id": "history_smoke",
        "root": "/grogu/datasets/hbhatia/3dfa_online_eval_50rollouts/video_deltam_history_smoke_v01",
        "plan": "instructions/eval_plans/video_deltam_history_smoke_v01.json",
        "status": "complete", "comparability": "smoke_only",
    },
    {
        "id": "proprio_ab_causal", "root": "/grogu/datasets/hbhatia/3dfa_online_eval_audit/video_deltam_proprio_ab_causal_v01",
        "plan": "instructions/eval_plans/video_deltam_proprio_ab_causal_v01.json",
        "status": "complete", "comparability": "single_task_diagnostic",
    },
    {
        "id": "proprio_ab_trainfaithful", "root": "/grogu/datasets/hbhatia/3dfa_online_eval_audit/video_deltam_proprio_ab_trainfaithful_v01",
        "plan": "instructions/eval_plans/video_deltam_proprio_ab_trainfaithful_v01.json",
        "status": "complete", "comparability": "single_task_diagnostic",
    },
    {
        "id": "proprio_parity_smoke", "root": "/grogu/datasets/hbhatia/3dfa_online_eval_audit/video_deltam_proprio_parity_smoke_v01",
        "plan": "instructions/eval_plans/video_deltam_proprio_parity_smoke_v01.json",
        "status": "complete", "comparability": "smoke_only",
    },
    {
        "id": "task_heldout_v02", "root": "/grogu/datasets/hbhatia/3dfa_online_eval_50rollouts/video_deltam_task_heldout_g1_g6_v02_visual_history",
        "plan": None, "status": "incomplete", "comparability": "intermediate_history_fix",
    },
    {
        "id": "task_heldout_v03", "root": "/grogu/datasets/hbhatia/3dfa_online_eval_50rollouts/video_deltam_task_heldout_g1_g6_v03_history_parity",
        "plan": "instructions/eval_plans/video_deltam_task_heldout_g1_g6_v01.json",
        "status": "incomplete", "comparability": "historical_parity_attempt",
    },
    {
        "id": "seen_base_ladder", "root": "/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts",
        "prefix": "video_deltam_checkpoint_ladder_v01_",
        "plan": "instructions/eval_plans/video_deltam_checkpoint_ladder_v01.json",
        "status": "complete_through_130k", "comparability": "training_style_seen_group",
    },
    {
        "id": "seen_base_best_84k", "root": "/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts/video_deltam_sbrs_best_v01",
        "plan": "instructions/eval_plans/video_deltam_sbrs_best_v01.json",
        "status": "complete", "comparability": "training_style_seen_group",
    },
    {
        "id": "seen_base_smoke_80k", "root": "/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts/video_deltam_sbrs_smoke_80k_v01",
        "plan": "instructions/eval_plans/video_deltam_sbrs_smoke_80k_v01.json",
        "status": "complete", "comparability": "training_style_seen_group",
    },
    {
        "id": "ulrs_g7_100rollouts", "root": "/grogu/datasets/hbhatia/3dfa_online_eval_100rollouts/ULRS_G7_video_deltam_100rollouts_v01",
        "plan": "instructions/eval_plans/ulrs_g7_video_deltam_100rollouts_v01.json",
        "status": "complete", "comparability": "fully_unknown_lab_g7",
    },
]


def campaign_dirs(spec: dict) -> list[Path]:
    root = Path(spec["root"])
    prefix = spec.get("prefix")
    return sorted(root.glob(f"{prefix}*")) if prefix else [root]


def collect(spec: dict) -> dict:
    cells: list[dict] = []
    roots = campaign_dirs(spec)
    for campaign_root in roots:
        for path in sorted(campaign_root.rglob("results_*.json")):
            if path.name.endswith((".manifest.json", ".progress.json")):
                continue
            try:
                payload = json.loads(path.read_text())
                task, metrics = next(iter(payload.items()))
                mean = float(metrics["mean"])
            except (OSError, ValueError, KeyError, StopIteration):
                continue
            relative = path.relative_to(campaign_root)
            # The ladder campaign shares a campaign-level label even though
            # the 110k and 140k checkpoint subdirectories are partial.
            result_status = spec["status"]
            if campaign_root.name.endswith(("s110000", "s140000")):
                result_status = "incomplete"
            # Standard layout is method/view/calibration/results_task.json.
            parts = relative.parts
            method = parts[0] if len(parts) >= 1 else None
            viewpoint = parts[1] if len(parts) >= 2 else None
            calibration = parts[2] if len(parts) >= 3 else None
            cells.append({
                "campaign_id": spec["id"], "campaign_status": spec["status"],
                "result_status": result_status,
                "comparability": spec["comparability"], "campaign_root": str(campaign_root),
                "relative_result_path": str(relative),
                "result_path": str(path), "manifest_path": str(path.with_suffix(".manifest.json")),
                "method": method, "viewpoint": viewpoint, "calibration": calibration,
                "task": task, "mean_success": mean,
            })
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for cell in cells:
        groups[(cell["campaign_root"], cell["method"], cell["viewpoint"], cell["calibration"])].append(cell)
    summaries = []
    for key, group in sorted(groups.items()):
        summaries.append({
            "campaign_root": key[0], "method": key[1], "viewpoint": key[2], "calibration": key[3],
            "completed_tasks": len(group),
            "macro_mean_success": sum(x["mean_success"] for x in group) / len(group),
            "task_means": {x["task"]: x["mean_success"] for x in group},
        })
    return {**spec, "raw_result_roots": [str(x) for x in roots], "cells": cells, "summaries": summaries}


def main() -> None:
    campaigns = [collect(spec) for spec in CAMPAIGNS]
    index = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": "Machine-readable index of Video-DeltaM raw online-evaluation JSON files and aggregates.",
        "campaigns": [{k: v for k, v in campaign.items() if k != "cells"} for campaign in campaigns],
    }
    task_metrics = {
        "schema_version": 1,
        "generated_at": index["generated_at"],
        "description": "One record per raw Video-DeltaM task result; use for plotting/re-aggregation.",
        "cells": [cell for campaign in campaigns for cell in campaign["cells"]],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "video_deltam_results_index.json").write_text(json.dumps(index, indent=2) + "\n")
    (OUT_DIR / "video_deltam_task_metrics.json").write_text(json.dumps(task_metrics, indent=2) + "\n")
    # CSV is deliberately flat so it opens directly in pandas, a spreadsheet,
    # or a plotting tool without a JSON-normalisation step.
    cells = task_metrics["cells"]
    with (OUT_DIR / "video_deltam_task_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]) if cells else [])
        writer.writeheader()
        writer.writerows(cells)


if __name__ == "__main__":
    main()
