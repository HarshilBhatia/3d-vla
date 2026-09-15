#!/usr/bin/env python3
"""Create the clean OOD-camera comparison table and figure.

The first four bars use the canonical G7, clean-extrinsics online-evaluation
artifacts.  The 2D run is intentionally displayed as a separately marked
reference: its only complete result uses the ordinary held-out G1--G6 group
mapping, not the new G7 camera.  Keeping that distinction visible prevents an
invalid G7 comparison while retaining the useful 2D baseline result.
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


@dataclass(frozen=True)
class Arm:
    name: str
    short_label: str
    training: str
    camera: str
    path: Path
    color: str
    hatch: str = ""

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/grogu/datasets/hbhatia/3dfa_online_eval_50rollouts"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    root = args.results_root
    arms = (
        Arm(
            "Clean 3D baseline",
            "Clean 3D",
            "Clean",
            "OOD G7",
            root / "clean" / "external",
            COLORS["baseline"],
        ),
        Arm(
            "R1a",
            "R1a",
            "Base + jitter",
            "OOD G7",
            root / "R1a_G7_clean" / "clean" / "external",
            COLORS["r1a"],
        ),
        Arm(
            "R1b",
            "R1b",
            "ΔM + base + jitter",
            "OOD G7",
            root / "R1b_G7_clean" / "clean" / "external",
            COLORS["r1b"],
        ),
        Arm(
            "R1c",
            "R1c",
            "ΔM + EE aux + base + jitter",
            "OOD G7",
            root / "R1c_G7_clean" / "clean" / "external",
            COLORS["r1c"],
        ),
        Arm(
            "2D baseline†",
            "2D†",
            "Clean",
            "Held-out G1–G6 group",
            root / "2d_siglip2_best72k" / "clean" / "all",
            COLORS["reference"],
            "///",
        ),
    )

    task_results = {arm.name: load_task_means(arm.path) for arm in arms}
    assert_shared_tasks(task_results)
    means = np.array([mean_success(task_results[arm.name]) for arm in arms])

    configure_theme()
    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    style_axis(ax, success_axis=True, y_lim=(0, 75), y_ticks=np.arange(0, 76, 15))

    x = np.arange(len(arms))
    bars = []
    for xpos, arm, mean in zip(x, arms, means):
        bars.append(
            ax.bar(
                xpos,
                100 * mean,
                width=0.68,
                color=arm.color,
                hatch=arm.hatch,
                edgecolor=COLORS["annotation"] if arm.hatch else arm.color,
                linewidth=0.9 if arm.hatch else 0,
                zorder=3,
            )[0]
        )
    for xpos, bar, mean in zip(x, bars, means):
        ax.annotate(
            f"{100 * mean:.1f}%",
            (xpos, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=COLORS["annotation"],
            fontweight="bold",
            fontsize=11,
        )

    ax.set_xticks(x, [arm.short_label for arm in arms])
    ax.set_ylabel("Mean task success rate")
    ax.set_title("Clean-extrinsics performance on an OOD camera", pad=14)
    add_protocol_note(
        ax,
        "3D bars: OOD G7 · clean test extrinsics · 13 tasks × 50 rollouts/task\n"
        "† 2D reference: held-out G1–G6 camera group, not G7",
        y=0.91,
        fontsize=8.7,
    )

    stem = args.output_dir / "g7_clean_model_comparison"
    png_path, svg_path = save_figure(fig, stem)
    plt.close(fig)

    table_path = stem.with_suffix(".md")
    table_lines = [
        "# OOD camera, clean-extrinsics comparison",
        "",
        "| Model | Training | Evaluation camera | Test miscal | Mean success |",
        "|---|---|---|---|---:|",
    ]
    for arm, mean in zip(arms, means):
        table_lines.append(
            f"| {arm.name} | {arm.training} | {arm.camera} | Clean | {100 * mean:.2f}% |"
        )
    table_lines.extend(
        [
            "",
            "All rows contain 13 task means. The four 3D rows are the G7 clean evaluation; "
            "the 2D row is marked separately because its available result uses the held-out "
            "G1–G6 group mapping rather than G7.",
            "",
        ]
    )
    write_markdown_table(table_path, table_lines)

    print("Summary (13-task mean success):")
    for arm, mean in zip(arms, means):
        print(f"{arm.name}: {100 * mean:.2f}% ({arm.camera})")
    print("Outputs:")
    print(png_path)
    print(svg_path)
    print(table_path)


if __name__ == "__main__":
    main()
