"""Each model's original checkpoint vs its best EMA checkpoint: task-mean success on IDLT task-heldout and the G7 extremes."""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.plotting import week_figures_dir

from plot_ablation_per_task import (EMA_MODELS, G7_ROT, G7_TRANS, GRID, IDLT_CAMPAIGNS, IDLT_LEVELS, MODELS, MUTED, R20,
                                    R100, TEXT, load, series)

OUT = week_figures_dir()


def mean_curve(cells, ids, levels, ntasks):
    xs, ys = [], []
    for i, (lvl, _) in enumerate(levels):
        v = series(cells, ids, lvl)
        if len(v) == ntasks:
            xs.append(i)
            ys.append(sum(v.values()) / ntasks)
    return xs, ys


def main():
    idlt = load(R100, IDLT_CAMPAIGNS)
    g7 = load(R20, ["g7_extreme_opposite_v01", "g7_rotation_only_v01", "g7_opposite_camera_sbrs_v01"])
    suites = [("IDLT task-heldout (13 × 100)", idlt, IDLT_LEVELS, 13, "miscal (°/cm)"),
              ("G7 opposing translation (6 × 20)", g7, G7_TRANS, 6, "cm"),
              ("G7 rotation about Z (6 × 20)", g7, G7_ROT, 6, "rotation")]
    fig, axes = plt.subplots(len(suites), len(MODELS), figsize=(2.9 * len(MODELS), 2.7 * len(suites)),
                             sharey=True, facecolor="#fcfcfb")
    for r, (sname, cells, levels, nt, xl) in enumerate(suites):
        base_xs, base_ys = mean_curve(cells, MODELS[-1][1], levels, nt)
        for c, ((label, ids, colour, marker), (elabel, eids, _, _)) in enumerate(zip(MODELS, EMA_MODELS)):
            ax = axes[r, c]
            ax.set_ylim(-3, 103)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=6.5)
            if c < len(MODELS) - 1:
                ax.plot(base_xs, base_ys, color="#8a8985", linewidth=1.2, linestyle=":", label="Base 160k (reference)")
            ox, oy = mean_curve(cells, ids, levels, nt)
            ex, ey = mean_curve(cells, eids, levels, nt)
            ax.plot(ox, oy, color=colour, linewidth=2, marker=marker, markersize=4, label="original checkpoint")
            ax.plot(ex, ey, color=colour, linewidth=2, linestyle="--", marker=marker, markersize=4,
                    markerfacecolor="#fcfcfb", label="best EMA checkpoint")
            ends = [(ys[-1], xs[-1]) for xs, ys in ((ox, oy), (ex, ey)) if xs]
            # The higher end value is labelled above its point, the lower one below, so labels never swap sides.
            for rank, (y, x) in enumerate(sorted(ends, reverse=True)):
                ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(4, 6 if rank == 0 else -11),
                            fontsize=7, color=TEXT)
            ax.set_xticks(range(len(levels)), [t for _, t in levels], fontsize=6.5)
            if r == 0:
                ax.set_title(f"{label}\nvs {elabel}", fontsize=8, color=TEXT)
            if c == 0:
                ax.set_ylabel(f"{sname}\nmean success %", fontsize=7.5, color=MUTED)
            ax.set_xlabel(xl, fontsize=6.5, color=MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Original vs best-EMA checkpoint per model (solid = original, dashed = EMA; label = rightmost value)",
                 fontsize=11, color=TEXT, y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "ema_vs_original.png", dpi=160, bbox_inches="tight")
    print("wrote", OUT / "ema_vs_original.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=OUT,
                    help="where to write the figures (default: this week's experiment figures dir)")
    OUT = ap.parse_args().output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    main()
