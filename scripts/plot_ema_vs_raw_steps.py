"""SBRS task-mean vs fine-tune step, raw vs EMA weights overlaid, one panel per EMA fine-tune."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_ema_ladders import GRID, MUTED, RUNS, TEXT, series

OUT = Path("docs/research/experiments/2026-09-29/figures")
RAW, EMA = "#8a8985", "#2a78d6"


def main():
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, sharey=True, facecolor="#fcfcfb")
    for ax, (run, label, family, start) in zip(axes.ravel(), RUNS):
        means = {}
        for weights, colour, ls, mk in (("raw", RAW, "--", "o"), ("ema", EMA, "-", "s")):
            pts = series(run, family, start, weights)
            means[weights] = {x: sum(v.values()) / len(v) for x, v in pts.items()}
            ax.plot(list(means[weights]), list(means[weights].values()), color=colour, linewidth=2.2, linestyle=ls,
                    marker=mk, markersize=5, label="raw weights" if weights == "raw" else "EMA weights")
        # At each step the higher of the two values is labelled above its point, the lower below, so labels never collide.
        for x in sorted(set(means["raw"]) | set(means["ema"])):
            vals = sorted(((means[w][x], w) for w in means if x in means[w]), reverse=True)
            for rank, (y, w) in enumerate(vals):
                ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 7 if rank == 0 else -13),
                            ha="center", fontsize=7.5, color=EMA if w == "ema" else MUTED)
        ax.set_ylim(45, 95)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_title(label, fontsize=9, color=TEXT)
    for ax in axes[:, 0]:
        ax.set_ylabel("SBRS mean over 6 tasks (%)", fontsize=8, color=MUTED)
    for ax in axes[-1]:
        ax.set_xlabel("fine-tune step (k); 0 = starting checkpoint", fontsize=8, color=MUTED)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("20k EMA fine-tunes: raw vs EMA weights over training (SBRS, 6 tasks × 20 rollouts)",
                 fontsize=11, color=TEXT, y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "ema_vs_raw_over_steps.png", dpi=170, bbox_inches="tight")
    print("wrote", OUT / "ema_vs_raw_over_steps.png")


if __name__ == "__main__":
    main()
