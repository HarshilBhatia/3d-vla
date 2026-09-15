#!/usr/bin/env python3
"""Plot external-camera test-miscalibration results with a clean-trained reference.

Reads the canonical per-task JSON results and produces PNG + SVG figures.  The
The two new training arms use 50 rollouts per task.  The clean-trained 3D
reference is the earlier camera-subset experiment (10 rollouts per task), which
is deliberately styled as a separate reference rather than a matched estimate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.plotting import (
    COLORS,
    assert_shared_tasks,
    configure_theme,
    load_task_means,
    mean_success,
    save_figure,
    style_axis,
)


CONDITIONS = (
    ("clean", "Clean", 0),
    ("random_external_2deg_2cm", "Test-2-Ext", 2),
    ("random_external_5deg_5cm", "Test-5-Ext", 5),
    ("random_external_10deg_10cm", "Test-10-Ext", 10),
)
ARMS = (
    ("base", "Base", COLORS["r1a"]),
    ("deltam_external", "DeltaM (external)", COLORS["deltam"]),
)
REFERENCE_PATHS = {
    "clean": "peract2_clean3d_camera_sweep/clean/external",
    "random_external_2deg_2cm": "peract2_clean3d_camera_sweep/2deg_2cm/external",
    "random_external_5deg_5cm": "peract2_clean3d_camera_subset/external/external",
    "random_external_10deg_10cm": "peract2_clean3d_camera_sweep/10deg_10cm/external",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("eval_logs/peract2_orbital_new_external_only_miscal"),
    )
    parser.add_argument(
        "--clean-reference-root",
        type=Path,
        default=Path("eval_logs"),
        help="Root containing the historical clean-trained camera-subset results.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    results: dict[str, dict[str, dict[str, float]]] = {}
    for arm, _, _ in ARMS:
        results[arm] = {
            condition: load_task_means(args.results_root / arm / condition / "external")
            for condition, _, _ in CONDITIONS
        }
    clean_reference = {
        condition: load_task_means(args.clean_reference_root / REFERENCE_PATHS[condition])
        for condition, _, _ in CONDITIONS
    }
    assert_shared_tasks(
        {
            **{
                f"{arm}/{condition}": results[arm][condition]
                for arm, _, _ in ARMS
                for condition, _, _ in CONDITIONS
            },
            **{f"clean_reference/{condition}": clean_reference[condition] for condition, _, _ in CONDITIONS},
        }
    )

    labels = [label for _, label, _ in CONDITIONS]
    x = np.arange(len(CONDITIONS))
    aggregate = {
        arm: np.array(
            [mean_success(results[arm][condition]) for condition, _, _ in CONDITIONS]
        )
        for arm, _, _ in ARMS
    }
    reference_values = np.array(
        [mean_success(clean_reference[condition]) for condition, _, _ in CONDITIONS]
    )

    configure_theme()
    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    style_axis(ax, y_lim=(45, 75), y_ticks=np.arange(45, 76, 5))

    ax.plot(
        x,
        100 * reference_values,
        marker="P",
        markersize=8,
        linewidth=2.5,
        linestyle="--",
        color=COLORS["baseline"],
        label="Clean train (10-rollout reference)",
        zorder=2,
    )

    for arm, label, color in ARMS:
        values = 100 * aggregate[arm]
        ax.plot(
            x,
            values,
            marker="o",
            markersize=8,
            linewidth=2.8,
            color=color,
            label=label,
            zorder=3,
        )
        for xpos, value in zip(x, values):
            offset = 1.8 if arm == "deltam_external" else -3.2
            ax.annotate(
                f"{value:.1f}",
                (xpos, value),
                xytext=(0, offset * 3),
                textcoords="offset points",
                ha="center",
                va="bottom" if offset > 0 else "top",
                color=color,
                fontweight="bold",
                fontsize=10,
            )

    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean task success rate (%)")
    ax.set_title("External-camera test miscalibration (wrists clean)", pad=14)
    ax.legend(frameon=False, loc="lower left", ncol=2, handlelength=2.2)

    stem = args.output_dir / "new_external_only_miscal_online_eval"
    png_path, svg_path = save_figure(fig, stem)
    plt.close(fig)

    print("Summary (mean success rate):")
    for arm, label, _ in ARMS:
        print(label, ", ".join(f"{name}={100 * value:.2f}%" for name, value in zip(labels, aggregate[arm])))
    print(
        "Clean train (10-rollout reference)",
        ", ".join(f"{name}={100 * value:.2f}%" for name, value in zip(labels, reference_values)),
    )
    print("Outputs:")
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
