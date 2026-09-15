#!/usr/bin/env python3
"""Plot archival G7 results with canonical calibration-realization labels.

All four 3D curves are evaluated on OOD G7, with clean extrinsics or a
fixed external-camera calibration realization held per condition. Directory
names below are historical and are intentionally not renamed.
The available 2D baseline is shown in a separate panel because it used the
ordinary held-out G1--G6 camera mapping rather than G7.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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
    ("Calibrated", "clean"),
    ("ext-r2deg-t2cm-v01", "2deg_2cm"),
    ("ext-r5deg-t5cm-v01", "5deg_5cm"),
    ("ext-r10deg-t10cm-v01", "10deg_10cm"),
    ("ext-r15deg-t15cm-v01", "15deg_15cm"),
)


@dataclass(frozen=True)
class Arm:
    name: str
    training: str
    roots: dict[str, str]
    color: str
    marker: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/grogu/datasets/hbhatia/3dfa_online_eval_50rollouts"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument(
        "--r1-only",
        action="store_true",
        help="Plot only R1a/R1b/R1c and omit the clean-3D and 2D references.",
    )
    args = parser.parse_args()

    all_arms = (
        Arm(
            "Calibrated-training 3D",
            "Calibrated training; no view alignment",
            {
                "clean": "clean/external",
                "2deg_2cm": "G7_random_2deg_2cm/external",
                "5deg_5cm": "G7_random_5deg_5cm/external",
                "10deg_10cm": "G7_random_10deg_10cm/external",
                "15deg_15cm": "G7_random_15deg_15cm/external",
            },
            COLORS["baseline"],
            "o",
        ),
        Arm(
            "R1a",
            "Group miscalibration + sampled miscalibration; no view alignment",
            {
                "clean": "R1a_G7_clean/clean/external",
                "2deg_2cm": "R1a_G7_random_2deg_2cm/2deg_2cm/external",
                "5deg_5cm": "R1a_G7_random_5deg_5cm/5deg_5cm/external",
                "10deg_10cm": "R1a_G7_random_10deg_10cm/10deg_10cm/external",
                "15deg_15cm": "R1a_G7_random_15deg_15cm/15deg_15cm/external",
            },
            COLORS["r1a"],
            "s",
        ),
        Arm(
            "R1b",
            "View alignment + group/sampled miscalibration",
            {
                "clean": "R1b_G7_clean/clean/external",
                "2deg_2cm": "R1b_G7_random_2deg_2cm/2deg_2cm/external",
                "5deg_5cm": "R1b_G7_random_5deg_5cm/5deg_5cm/external",
                "10deg_10cm": "R1b_G7_random_10deg_10cm/10deg_10cm/external",
                "15deg_15cm": "R1b_G7_random_15deg_15cm/15deg_15cm/external",
            },
            COLORS["r1b"],
            "D",
        ),
        Arm(
            "R1c",
            "View alignment + EE auxiliary supervision + group/sampled miscalibration",
            {
                "clean": "R1c_G7_clean/clean/external",
                "2deg_2cm": "R1c_G7_random_2deg_2cm/2deg_2cm/external",
                "5deg_5cm": "R1c_G7_random_5deg_5cm/5deg_5cm/external",
                "10deg_10cm": "R1c_G7_random_10deg_10cm/10deg_10cm/external",
                "15deg_15cm": "R1c_G7_random_15deg_15cm/15deg_15cm/external",
            },
            COLORS["r1c"],
            "^",
        ),
    )
    arms = tuple(arm for arm in all_arms if args.r1_only is False or arm.name.startswith("R1"))
    root = args.results_root
    results: dict[str, list[float]] = {}
    for arm in arms:
        values = []
        for _, key in CONDITIONS:
            task_values = load_task_means(root / arm.roots[key])
            values.append(task_values)
        results[arm.name] = values

    two_d = None if args.r1_only else load_task_means(root / "2d_siglip2_best72k/clean/all")
    assert_shared_tasks(
        {
            **{f"{arm.name}/{key}": results[arm.name][index] for arm in arms for index, (_, key) in enumerate(CONDITIONS)},
            **({"2D": two_d} if two_d is not None else {}),
        }
    )
    means = {arm.name: [mean_success(task_means) for task_means in results[arm.name]] for arm in arms}
    two_d_mean = mean_success(two_d) if two_d is not None else None

    configure_theme()
    if args.r1_only:
        fig, ax = plt.subplots(figsize=(9.6, 5.7), constrained_layout=True)
        ax_2d = None
    else:
        fig, (ax, ax_2d) = plt.subplots(
            1,
            2,
            figsize=(12.4, 5.7),
            gridspec_kw={"width_ratios": [4.7, 1.15]},
            constrained_layout=True,
        )
    fig.patch.set_facecolor("white")
    for panel in (ax,) if ax_2d is None else (ax, ax_2d):
        style_axis(panel, success_axis=True, y_lim=(0, 75), y_ticks=np.arange(0, 76, 15))

    x = np.arange(len(CONDITIONS))
    for arm in arms:
        y = 100 * np.array(means[arm.name])
        ax.plot(
            x,
            y,
            color=arm.color,
            marker=arm.marker,
            markersize=7,
            linewidth=2.5,
            label=arm.name,
            zorder=3,
        )
        ax.annotate(
            f"{y[-1]:.1f}%",
            (x[-1], y[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=arm.color,
            fontweight="bold",
            fontsize=9.5,
        )
    ax.set_xticks(x, [name for name, _ in CONDITIONS])
    ax.set_ylabel("Mean task success rate (%)")
    ax.set_title(
        "G7 fully OOD: R1 models across external-camera calibration realizations"
        if args.r1_only
        else "G7 fully OOD: calibrated and external-camera calibration realizations",
        pad=13,
    )
    ax.legend(frameon=False, loc="lower left", ncol=2, handlelength=2.2)
    add_protocol_note(
        ax,
        "External-camera-only calibration realization; wrists calibrated\n13 tasks × 50 rollouts/task",
    )

    if ax_2d is not None:
        bar = ax_2d.bar(
            0,
            100 * two_d_mean,
            width=0.62,
            color=COLORS["reference"],
            hatch="///",
            edgecolor="#35414C",
            linewidth=0.9,
            zorder=3,
        )[0]
        ax_2d.annotate(
            f"{100 * two_d_mean:.1f}%",
            (0, bar.get_height()),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            fontsize=10.5,
        )
        ax_2d.set_xticks([0], ["2D†\nCalibrated"])
        ax_2d.set_yticklabels([])
        ax_2d.tick_params(axis="y", length=0)
        ax_2d.set_title("2D reference", pad=13)
        add_protocol_note(
            ax_2d,
            "† Held-out\nG1–G6 group\n(not G7)",
            x=0.5,
            y=0.96,
            ha="center",
            fontsize=8.6,
        )

    stem = args.output_dir / (
        "g7_r1_eval_miscal_comparison" if args.r1_only else "g7_eval_miscal_model_comparison"
    )
    png_path, svg_path = save_figure(fig, stem)
    plt.close(fig)

    table = [
        "# OOD G7 external-camera miscalibration comparison",
        "",
        "| Model | Training | Camera | Calibrated | External: 2° + 2 cm | External: 5° + 5 cm | External: 10° + 10 cm | External: 15° + 15 cm |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        values = means[arm.name]
        table.append(
            f"| {arm.name} | {arm.training} | OOD G7 | "
            + " | ".join(f"{100 * value:.2f}%" for value in values)
            + " |"
        )
    if two_d_mean is not None:
        table.append(
            f"| 2D baseline† | Calibrated training; no view alignment | Held-out G1–G6 | {100 * two_d_mean:.2f}% | — | — | — | — |",
        )
    table.extend(
        [
            "",
            "All cells contain the mean over 13 bimanual tasks and 50 rollouts/task. "
            "Each external-camera condition is a newly sampled geometric miscalibration held fixed per evaluation condition; wrist cameras remain calibrated.",
            "",
        ]
    )
    table_path = stem.with_suffix(".md")
    write_markdown_table(table_path, table)

    print("G7 13-task means:")
    for arm in arms:
        print(arm.name, ", ".join(f"{name}={100 * value:.2f}%" for (name, _), value in zip(CONDITIONS, means[arm.name])))
    if two_d_mean is not None:
        print(f"2D reference (not G7): Clean={100 * two_d_mean:.2f}%")
    print("Outputs:")
    print(png_path)
    print(svg_path)
    print(table_path)


if __name__ == "__main__":
    main()
