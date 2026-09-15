#!/usr/bin/env python3
"""Plot the best ULRS G7 result per model and calibration realization.

The Base and DeltaM curves are re-aggregated from their canonical result JSONs.
Video-DeltaM is re-aggregated from its exported task-level CSV.  The selected
checkpoint may differ by calibration level; the legend intentionally calls
this out rather than presenting it as one checkpoint trajectory.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.plotting import (
    COLORS,
    add_protocol_note,
    assert_shared_tasks,
    configure_theme,
    load_task_means,
    mean_success,
    save_figure,
    style_axis,
    write_markdown_table,
)


CONDITIONS = (
    ("calibrated", "Clean"),
    ("ext-r2deg-t2cm-v01", "External\n2° / 2 cm"),
    ("ext-r5deg-t5cm-v01", "External\n5° / 5 cm"),
    ("ext-r10deg-t10cm-v01", "External\n10° / 10 cm"),
)
ULRS_METHODS = {
    "Base": ("base_s140k", "base_s160k"),
    "DeltaM": ("deltam_s140k", "deltam_s160k", "deltam_s180k"),
}
VIDEO_METHODS = (
    "video_deltam_best_s084k",
    "video_deltam_s100k",
    "video_deltam_s130k",
)
VIDEO_CALIBRATIONS = {
    "calibrated": "calibrated",
    "ext-r2deg-t2cm-v01": "ext-r2deg-t2cm-v01",
    "ext-r5deg-t5cm-v01": "ext-r5deg-t5cm-v01",
    "ext-r10deg-t10cm-v01": "ext-r10deg-t10cm-v01",
}


def load_video_results(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Load decision-grade ULRS Video-DeltaM task means from the export."""
    results: dict[str, dict[str, dict[str, float]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["campaign_id"] != "ulrs_g7_100rollouts"
                or row["result_status"] != "complete"
                or row["method"] not in VIDEO_METHODS
            ):
                continue
            method = row["method"]
            calibration = row["calibration"]
            if calibration not in VIDEO_CALIBRATIONS.values():
                continue
            task = row["task"]
            slot = results.setdefault(method, {}).setdefault(calibration, {})
            if task in slot:
                raise RuntimeError(f"Duplicate Video-DeltaM result: {method}/{calibration}/{task}")
            slot[task] = float(row["mean_success"])
    for method in VIDEO_METHODS:
        if method not in results:
            raise RuntimeError(f"Missing Video-DeltaM method: {method}")
        for calibration, _ in CONDITIONS:
            if calibration not in results[method]:
                raise RuntimeError(f"Missing Video-DeltaM condition: {method}/{calibration}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ulrs-root",
        type=Path,
        default=Path("/grogu/datasets/hbhatia/3dfa_online_eval_100rollouts/ULRS_G7_100rollouts_v01"),
    )
    parser.add_argument("--video-metrics", type=Path, default=Path("docs/results/video_deltam_task_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval/interp"))
    parser.add_argument("--base-peract2", type=float, default=85.0)
    parser.add_argument("--camvar-peract2", type=float, default=79.2)
    args = parser.parse_args()

    raw: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for label, methods in ULRS_METHODS.items():
        raw[label] = {}
        for method in methods:
            raw[label][method] = {
                calibration: load_task_means(args.ulrs_root / method / "g7_fully_unknown_lab" / calibration)
                for calibration, _ in CONDITIONS
            }
    raw["Video-DeltaM"] = load_video_results(args.video_metrics)

    # Every checkpoint-condition cell must cover exactly the same 13 tasks.
    assert_shared_tasks(
        {
            f"{label}/{method}/{calibration}": task_means
            for label, by_method in raw.items()
            for method, by_condition in by_method.items()
            for calibration, task_means in by_condition.items()
        }
    )

    best_values: dict[str, list[float]] = {}
    selected: dict[str, list[str]] = {}
    for label, by_method in raw.items():
        best_values[label], selected[label] = [], []
        for calibration, _ in CONDITIONS:
            winner, value = max(
                ((method, mean_success(by_condition[calibration])) for method, by_condition in by_method.items()),
                key=lambda item: item[1],
            )
            best_values[label].append(value)
            selected[label].append(winner)

    configure_theme()
    fig, ax = plt.subplots(figsize=(9.2, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    style_axis(ax, success_axis=True, y_lim=(50, 90), y_ticks=np.arange(50, 91, 5))
    x = np.arange(len(CONDITIONS))
    curves = (
        ("Base", COLORS["r1a"], "o"),
        ("DeltaM", COLORS["deltam"], "s"),
        ("Video-DeltaM", "#6A3D9A", "D"),
    )
    for label, color, marker in curves:
        values = 100 * np.asarray(best_values[label])
        ax.plot(x, values, marker=marker, color=color, linewidth=2.8, markersize=7.5, label=label, zorder=3)
        for xpos, value in zip(x, values):
            ax.annotate(f"{value:.1f}", (xpos, value), xytext=(0, 8), textcoords="offset points", ha="center", color=color, fontweight="bold", fontsize=9)

    ax.axhline(args.base_peract2, color="#2F4858", linewidth=2.1, linestyle="--", label=f"Base PerAct2 ({args.base_peract2:.1f}%)")
    ax.axhline(args.camvar_peract2, color=COLORS["reference"], linewidth=2.1, linestyle=(0, (4, 2)), label=f"Cam-var PerAct2 ({args.camvar_peract2:.1f}%)")
    ax.set_xticks(x, [label for _, label in CONDITIONS])
    ax.set_ylabel("Mean task success rate")
    ax.set_title("ULRS G7: external-camera calibration robustness")
    add_protocol_note(ax, "G7 fully unknown lab · 13 tasks × 100 rollouts/task\nExternal cameras perturbed; wrist cameras calibrated\nPoints select the best available checkpoint per condition.")
    ax.legend(loc="lower left", frameon=False, ncol=2, fontsize=9.2)

    stem = args.output_dir / "ulrs_g7_best_model_comparison"
    png_path, svg_path = save_figure(fig, stem)
    plt.close(fig)

    lines = [
        "# ULRS G7 — best model result per calibration condition",
        "",
        "| Model | Clean | External 2° / 2 cm | External 5° / 5 cm | External 10° / 10 cm |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, _, _ in curves:
        cells = [f"{100 * value:.1f}% ({method})" for value, method in zip(best_values[label], selected[label])]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.extend([
        f"| Base PerAct2 reference | {args.base_peract2:.1f}% | {args.base_peract2:.1f}% | {args.base_peract2:.1f}% | {args.base_peract2:.1f}% |",
        f"| Cam-var PerAct2 reference | {args.camvar_peract2:.1f}% | {args.camvar_peract2:.1f}% | {args.camvar_peract2:.1f}% | {args.camvar_peract2:.1f}% |",
        "",
        "Base/DeltaM points are re-aggregated from the ULRS raw task JSONs; Video-DeltaM points are re-aggregated from the decision-grade ULRS rows in `docs/results/video_deltam_task_metrics.csv`. The two PerAct2 references were supplied as horizontal reference levels and are not asserted to share this G7 external-camera protocol.",
    ])
    markdown_path = write_markdown_table(stem.with_suffix(".md"), lines)

    print("Best mean success rates and selected checkpoints:")
    for label, _, _ in curves:
        print(label, ", ".join(f"{100 * value:.1f}% ({method})" for value, method in zip(best_values[label], selected[label])))
    print(f"Base PerAct2 reference: {args.base_peract2:.1f}%")
    print(f"Cam-var PerAct2 reference: {args.camvar_peract2:.1f}%")
    print(png_path)
    print(svg_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
