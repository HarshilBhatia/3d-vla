"""One figure per model: original vs best-EMA checkpoint on every IDLT task-heldout and G7 extreme task."""
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
SLUG = ["vidpooled", "randhist", "noviewalign", "regtest_a", "regtest_c", "base"]


def curve(cells, ids, levels, task, ntasks):
    xs, ys = [], []
    for i, (lvl, _) in enumerate(levels):
        v = series(cells, ids, lvl)
        y = (sum(v.values()) / ntasks if len(v) == ntasks else None) if task == "mean" else v.get(task)
        if y is not None:
            xs.append(i)
            ys.append(y)
    return xs, ys


def main():
    idlt = load(R100, IDLT_CAMPAIGNS)
    g7 = load(R20, ["g7_extreme_opposite_v01", "g7_rotation_only_v01", "g7_opposite_camera_sbrs_v01"])
    t13 = sorted({t for (m, c), d in idlt.items() if c == "calibrated" for t in d})
    t6 = sorted({t for (m, c), d in g7.items() if c == "opposite-200cm" for t in d})
    # Rows: IDLT (two rows of 7 panels), G7 translation, G7 rotation; the first panel of each block is the task mean.
    blocks = [("IDLT", idlt, IDLT_LEVELS, ["mean"] + t13[:6], 13), ("IDLT", idlt, IDLT_LEVELS, t13[6:], 13),
              ("G7 translation", g7, G7_TRANS, ["mean"] + t6, 6), ("G7 rotation", g7, G7_ROT, ["mean"] + t6, 6)]
    OUT.mkdir(parents=True, exist_ok=True)
    for slug, (label, ids, colour, marker), (elabel, eids, _, _) in zip(SLUG, MODELS, EMA_MODELS):
        fig, axes = plt.subplots(len(blocks), 7, figsize=(19, 2.55 * len(blocks)), sharey=True, facecolor="#fcfcfb")
        for r, (suite, cells, levels, tasks, nt) in enumerate(blocks):
            for c in range(7):
                ax = axes[r, c]
                if c >= len(tasks):
                    ax.axis("off")
                    continue
                task = tasks[c]
                ax.set_ylim(-3, 103)
                ax.grid(axis="y", color=GRID, linewidth=0.8)
                ax.set_axisbelow(True)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                ax.tick_params(colors=MUTED, labelsize=6)
                for pid, style, face, name in ((ids, "-", colour, "original"), (eids, "--", "#fcfcfb", "EMA")):
                    xs, ys = curve(cells, pid, levels, task, nt)
                    ax.plot(xs, ys, color=colour, linewidth=2, linestyle=style, marker=marker, markersize=3.8,
                            markerfacecolor=face, label=f"{name}")
                ax.set_xticks(range(len(levels)), [t for _, t in levels], fontsize=6)
                ax.set_title(f"{suite}: {'task mean' if task == 'mean' else task}", fontsize=7.5, color=TEXT,
                             fontweight="bold" if task == "mean" else "normal")
            axes[r, 0].set_ylabel("success %", fontsize=7.5, color=MUTED)
        handles = [plt.Line2D([], [], color=colour, linewidth=2, marker=marker, label=f"original: {label}"),
                   plt.Line2D([], [], color=colour, linewidth=2, linestyle="--", marker=marker, markerfacecolor="#fcfcfb",
                              label=f"best EMA: {elabel}")]
        fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(f"{label} vs {elabel}: IDLT task-heldout (13 × 100) and G7 extremes (6 × 20), per task",
                     fontsize=11, color=TEXT, y=1.03)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(OUT / f"ema_per_model_{slug}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", OUT / f"ema_per_model_{slug}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=OUT,
                    help="where to write the figures (default: this week's experiment figures dir)")
    OUT = ap.parse_args().output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    main()
