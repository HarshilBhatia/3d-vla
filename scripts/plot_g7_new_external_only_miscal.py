#!/usr/bin/env python3
"""Plot the matched OOD-G7 external-camera test-miscalibration comparison.

All curves use the same 13 bimanual tasks, OOD G7 camera geometry, 50 rollouts
per task, and external-only test perturbation (orbital cameras only; wrists
clean).  The clean-trained reference is read from the canonical historical G7
artifacts; the two new arms are read from the dedicated G7 array output.
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
    fit_clean_anchored_phase_trend,
    load_task_means,
    mean_success,
    rope_phase_bound,
    save_figure,
    style_axis,
    write_markdown_table,
)


CONDITIONS = (
    ("calibrated", "Calibrated"),
    ("ext-r2deg-t2cm-v01", "ext-r2deg-t2cm-v01"),
    ("ext-r5deg-t5cm-v01", "ext-r5deg-t5cm-v01"),
    ("ext-r10deg-t10cm-v01", "ext-r10deg-t10cm-v01"),
)
CURVES = (
    ("clean_train", "Calibrated-training 3D", COLORS["baseline"], "P", "--"),
    ("base", "Miscalibration-trained control", COLORS["r1a"], "o", "-"),
    ("deltam_external", "View alignment (external)", COLORS["deltam"], "s", "-"),
)
CLEAN_G7_PATHS = {
    "calibrated": "clean/external",
    "ext-r2deg-t2cm-v01": "G7_random_2deg_2cm/external",
    "ext-r5deg-t5cm-v01": "G7_random_5deg_5cm/external",
    "ext-r10deg-t10cm-v01": "G7_random_10deg_10cm/external",
}
HISTORICAL_NEW_PATHS = {
    "calibrated": "clean", "ext-r2deg-t2cm-v01": "2deg_2cm",
    "ext-r5deg-t5cm-v01": "5deg_5cm", "ext-r10deg-t10cm-v01": "10deg_10cm",
}
# The clean 3D checkpoint's workspace_normalizer has max absolute coordinate
# extent (0.896, 0.704, 1.579) m, so this is the norm of its farthest corner.
WORKSPACE_RADIUS_M = 1.946760703673942
# In RotaryPositionEncoding3D, the largest div_term is exp(0) = 1 rad/m.
ROPE_OMEGA_MAX = 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--new-results-root",
        type=Path,
        default=Path("/grogu/datasets/hbhatia/3dfa_online_eval_50rollouts/new_external_only_G7"),
    )
    parser.add_argument(
        "--clean-results-root",
        type=Path,
        default=Path("/grogu/datasets/hbhatia/3dfa_online_eval_50rollouts"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    results: dict[str, dict[str, dict[str, float]]] = {
        "clean_train": {
            condition: load_task_means(args.clean_results_root / CLEAN_G7_PATHS[condition])
            for condition, _ in CONDITIONS
        },
        **{
            arm: {
                condition: load_task_means(args.new_results_root / arm / HISTORICAL_NEW_PATHS[condition] / "external")
                for condition, _ in CONDITIONS
            }
            for arm in ("base", "deltam_external")
        },
    }
    assert_shared_tasks(
        {
            f"{arm}/{condition}": results[arm][condition]
            for arm, *_ in CURVES
            for condition, _ in CONDITIONS
        }
    )
    means = {
        arm: np.array([mean_success(results[arm][condition]) for condition, _ in CONDITIONS])
        for arm, *_ in CURVES
    }
    phase_bound = rope_phase_bound(
        [0.0, 2.0, 5.0, 10.0],
        [0.0, 0.02, 0.05, 0.10],
        rho_m=WORKSPACE_RADIUS_M,
        omega_max=ROPE_OMEGA_MAX,
    )
    phase_trend, phase_scale = fit_clean_anchored_phase_trend(phase_bound, means["base"])

    configure_theme()
    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    fig.patch.set_facecolor("white")
    style_axis(ax, success_axis=True, y_lim=(50, 70), y_ticks=np.arange(50, 71, 5))

    x = np.arange(len(CONDITIONS))
    for arm, label, color, marker, linestyle in CURVES:
        ax.plot(
            x,
            100 * means[arm],
            color=color,
            marker=marker,
            markersize=8,
            linewidth=2.7,
            linestyle=linestyle,
            label=label,
            zorder=3,
        )
    ax.plot(
        x,
        100 * phase_trend,
        color=COLORS["reference"],
        marker="x",
        markersize=8,
        markeredgewidth=2,
        linewidth=2.2,
        linestyle=":",
        label=f"Scaled RoPE phase bound (control fit: {100 * phase_scale:.1f} pp/rad)",
        zorder=2,
    )

    ax.set_xticks(x, [label for _, label in CONDITIONS])
    ax.set_ylabel("Mean task success rate (%)")
    ax.set_title("G7 fully OOD: external-camera calibration realizations (wrists calibrated)", pad=14)
    ax.legend(frameon=False, loc="lower left", ncol=2, handlelength=2.3)

    stem = args.output_dir / "g7_new_external_only_miscal"
    png_path, svg_path = save_figure(fig, stem)
    plt.close(fig)

    table = [
        "# OOD G7 external-camera-only miscalibration",
        "",
        "| Model | Calibrated | ext-r2deg-t2cm-v01 | ext-r5deg-t5cm-v01 | ext-r10deg-t10cm-v01 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, label, *_ in CURVES:
        table.append(f"| {label} | " + " | ".join(f"{100 * value:.2f}%" for value in means[arm]) + " |")
    table.append(
        "| RoPE phase bound $s$ (rad) | "
        + " | ".join(f"{value:.3f}" for value in phase_bound)
        + " |"
    )
    table.append(
        "| Scaled phase-bound trend (control fit) | "
        + " | ".join(f"{100 * value:.2f}%" for value in phase_trend)
        + " |"
    )
    table.extend(
        [
            "",
            "All model values are the mean over 13 bimanual tasks and 50 rollouts/task on OOD G7. "
            "Each condition names one fixed materialized calibration realization. Only orbital_left and orbital_right receive non-identity deltas; wrist cameras remain calibrated. "
            "The dotted curve is not an independent prediction: it is the phase bound "
            r"$s=\omega_{max}(2\sin(\alpha/2)\rho+\tau)$ mapped into success units by a "
            "non-negative least-squares slope fitted once to the control curve, anchored at the calibrated condition. "
            fr"Here $\omega_{{max}}={ROPE_OMEGA_MAX:.0f}$ rad/m and $\rho={WORKSPACE_RADIUS_M:.3f}$ m.",
            "",
        ]
    )
    table_path = stem.with_suffix(".md")
    write_markdown_table(table_path, table)

    print("OOD G7 13-task means:")
    for arm, label, *_ in CURVES:
        print(label, ", ".join(f"{condition_label}={100 * value:.2f}%" for (_, condition_label), value in zip(CONDITIONS, means[arm])))
    print("RoPE phase bound (rad):", ", ".join(f"{value:.3f}" for value in phase_bound))
    print(f"Scaled phase-bound fit: {100 * phase_scale:.2f} pp/rad")
    print("Outputs:")
    print(png_path)
    print(svg_path)
    print(table_path)


if __name__ == "__main__":
    main()
