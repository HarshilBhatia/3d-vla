"""Plot per-task G7 extreme translation/rotation results in one PNG.

The figure has six subplots (one per SBRS task).  Each subplot contains all
available methods: solid lines are opposing translations at 100/200/500 cm;
dashed lines are rotation-only results at 20/90/180 degrees.  The 270-degree
condition is intentionally omitted from this figure.

Only PNG output is produced.  The source values are read from the canonical
result JSON + manifest files on /grogu; no values are hard-coded here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts")
OUT = Path("docs/research/results/figures/g7_extreme_task_graphs.png")

TRANSLATION_CAMPAIGN = "g7_extreme_opposite_v01"
ROTATION_CAMPAIGN = "g7_rotation_only_v01"
TRANSLATION_EXTRA_CAMPAIGNS = [
    "g7_extreme_opposite_rope3d_v01",
    "g7_extreme_opposite_videopooled_direct_v01",
]
ROTATION_EXTRA_CAMPAIGNS = [
    "g7_rotation_only_rope3d_v01",
    "g7_rotation_only_videopooled_direct_v01",
]
TRANSLATION_LEVELS = ["g7-opposite-100cm-v01", "g7-opposite-200cm-v01", "g7-opposite-500cm-v01"]
ROTATION_LEVELS = ["g7-rot-20deg-v01", "g7-rot-90deg-v01", "g7-rot-180deg-v01"]
TASKS = [
    "bimanual_push_box",
    "bimanual_lift_ball",
    "bimanual_pick_laptop",
    "bimanual_pick_plate",
    "bimanual_straighten_rope",
    "bimanual_handover_item_easy",
]

METHOD_LABELS = {
    "base_s160k": "Base",
    "deltam_s140k": "DeltaM",
    "video_pooled_s084k": "VideoPooled-Refine",
    "video_fullpatch_best": "FullPatch Video",
    "direct_fullpatch_rope3d": "Direct Rope3D",
    "refine_fullpatch_rope3d": "Refine Rope3D",
    "video_pooled_direct": "Direct VideoPooled",
}
METHOD_ORDER = list(METHOD_LABELS.values())
COLORS = {
    "Base": "#4C78A8",
    "DeltaM": "#F58518",
    "VideoPooled-Refine": "#54A24B",
    "FullPatch Video": "#E45756",
    "Direct Rope3D": "#B279A2",
    "Refine Rope3D": "#FF9DA6",
    "Direct VideoPooled": "#72B7B2",
}


def load_campaign(campaign: str, levels: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    """Return method -> task -> calibration -> success rate."""
    root = ROOT / campaign
    if not root.is_dir():
        raise FileNotFoundError(root)
    values: dict[str, dict[str, dict[str, float]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for result_path in sorted(root.glob("**/results_*.json")):
        if result_path.name.endswith(".manifest.json"):
            continue
        manifest_path = result_path.with_name(result_path.name.replace(".json", ".manifest.json"))
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest for {result_path}")
        payload = json.loads(result_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        method_id = result_path.parts[-4]
        if method_id not in METHOD_LABELS:
            raise ValueError(f"Unknown method directory {method_id!r} in {result_path}")
        calibration_id = manifest["calibration_id"]
        if calibration_id not in levels:
            continue
        for task, result in payload.items():
            if task not in TASKS:
                continue
            if not isinstance(result, dict) or not isinstance(result.get("mean"), (int, float)):
                raise ValueError(f"Malformed result in {result_path}")
            key = (method_id, task, calibration_id)
            if key in seen:
                raise ValueError(f"Duplicate result cell {key} in {campaign}")
            seen.add(key)
            values.setdefault(METHOD_LABELS[method_id], {}).setdefault(task, {})[calibration_id] = float(result["mean"])
    return values


def merge_campaigns(campaigns: list[str], levels: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    merged: dict[str, dict[str, dict[str, float]]] = {}
    for campaign in campaigns:
        values = load_campaign(campaign, levels)
        for method, task_values in values.items():
            for task, level_values in task_values.items():
                destination = merged.setdefault(method, {}).setdefault(task, {})
                duplicate = set(destination).intersection(level_values)
                if duplicate:
                    raise ValueError(f"Duplicate merged cells for {method}/{task}: {sorted(duplicate)}")
                destination.update(level_values)
    return merged


def main() -> None:
    translation = merge_campaigns([TRANSLATION_CAMPAIGN, *TRANSLATION_EXTRA_CAMPAIGNS], TRANSLATION_LEVELS)
    rotation = merge_campaigns([ROTATION_CAMPAIGN, *ROTATION_EXTRA_CAMPAIGNS], ROTATION_LEVELS)
    available_tasks = set().union(*(set(v) for v in translation.values()), *(set(v) for v in rotation.values()))
    if available_tasks != set(TASKS):
        raise ValueError(f"Expected exactly six SBRS tasks; found {sorted(available_tasks)}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    x = np.array([0, 1, 2, 4, 5, 6], dtype=float)
    for ax, task in zip(axes.flat, TASKS):
        for method in METHOD_ORDER:
            tvals = [translation.get(method, {}).get(task, {}).get(level, np.nan) * 100 for level in TRANSLATION_LEVELS]
            rvals = [rotation.get(method, {}).get(task, {}).get(level, np.nan) * 100 for level in ROTATION_LEVELS]
            color = COLORS[method]
            ax.plot(x[:3], tvals, color=color, marker="o", linewidth=2, markersize=4, label=method)
            ax.plot(x[3:], rvals, color=color, marker="s", linestyle="--", linewidth=1.7, markersize=4)
        ax.axvline(3, color="#999999", linewidth=0.8)
        ax.set_xticks(x, ["100", "200", "500", "20", "90", "180"])
        ax.set_ylim(0, 105)
        ax.set_title(task.replace("bimanual_", "").replace("_", " ").title(), fontsize=12)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlabel("Translation (cm)  |  Rotation (degrees)", fontsize=9)
    axes[0, 0].set_ylabel("Success (%)")
    axes[1, 0].set_ylabel("Success (%)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(
        "G7 unseen-camera extreme calibration: task-level success\n"
        "Solid = opposing translation (100/200/500 cm); dashed = rotation-only (20/90/180°); 6 tasks × 20 rollouts",
        fontsize=15,
        y=1.045,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", format="png")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
