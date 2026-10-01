"""IDLT task-heldout task-mean success vs miscalibration for the best EMA checkpoints of the history models and Base."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_ablation_per_task import GRID, IDLT_CAMPAIGNS, MUTED, R100, TEXT, load, series

OUT = Path("docs/research/experiments/2026-09-29/figures")
LEVELS = [("calibrated", "0"), ("e10deg-t10cm", "10"), ("e15deg-t15cm", "15"), ("e20deg-t20cm", "20"),
          ("e30deg-t30cm", "30"), ("e40deg-t40cm", "40"), ("e50deg-t50cm", "50"), ("e60deg-t60cm", "60")]
# Validated categorical slots 1-3 for the history models; Base is the neutral reference.
MODELS = [("VideoPooled EMA 15k", "ema20k_vidpooled_s15k_ema", "#2a78d6", "o", "-"),
          ("Random history EMA 15k", "ema20k_randhist_s15k_ema", "#eb6834", "s", "-"),
          ("No delta_M EMA 15k", "ema20k_noviewalign_s15k_ema", "#1baf7a", "^", "-"),
          ("Base EMA 20k", "ema20k_base_s20k_ema", "#8a8985", "x", "--")]


def main():
    cells = load(R100, IDLT_CAMPAIGNS)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    rows = []
    xs = list(range(len(LEVELS)))
    for label, mid, colour, marker, ls in MODELS:
        ys = []
        for lvl, _ in LEVELS:
            v = series(cells, [mid], lvl)
            ys.append(sum(v.values()) / len(v) if len(v) == 13 else float("nan"))
            rows.append([label, lvl, round(ys[-1], 1)])
        ax.plot(xs, ys, color=colour, linewidth=2, linestyle=ls, marker=marker, markersize=6, label=label)
        ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(8, -3), fontsize=8.5, color=TEXT)
    ax.set_xticks(xs, [t for _, t in LEVELS])
    ax.set_xlim(-0.3, len(LEVELS) - 0.4)
    ax.set_ylim(30, 90)
    ax.set_xlabel("external-camera miscalibration (rotation ° / translation cm)", fontsize=9, color=MUTED)
    ax.set_ylabel("mean success over 13 tasks (%)", fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.set_title("IDLT task-heldout, best EMA checkpoints (13 tasks × 100 rollouts)", fontsize=10.5, color=TEXT)
    fig.tight_layout()
    fig.savefig(OUT / "ema_idlt_mean.png", dpi=180, bbox_inches="tight")
    with open(OUT / "ema_idlt_mean.csv", "w", newline="") as f:
        csv.writer(f).writerows([["model", "condition", "success_pct"]] + rows)
    print("wrote", OUT / "ema_idlt_mean.png")


if __name__ == "__main__":
    main()
