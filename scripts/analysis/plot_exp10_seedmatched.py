#!/usr/bin/env python3
"""Exp 10 seed-matched 100-episode eval: handover success vs camera miscalibration for base, base - proprio, pooled hist + deltaM, and its no-proprio variant."""
import re, subprocess, sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = [("exp8d_s50k_ema", "Base model (8d)", "#2a78d6"),
          ("exp10i_noproprio_s50k_ema", "Base − proprio (10i)", "#eb6834"),
          ("exp10f_pooled_refine_s100k_ema", "Pooled hist + δM (10f)", "#1baf7a"),
          ("exp10h_pooled_refine_noproprio_s100k_ema", "Pooled hist + δM − proprio (10h)", "#eda100")]
COLS = ["G1\ncal", "G4\ncal", "10/10", "15/15", "20/20", "30/30", "40/40", "50/50", "60/60"]


def pooled(method):
    out = subprocess.run([sys.executable, "scripts/analysis/aggregate_sharded_eval.py", method, "11"], capture_output=True, text=True).stdout
    vals = [tuple(map(int, m.groups())) for m in re.finditer(r"=(\d+)/(\d+)", out)]
    return [100 * s / n if n else None for s, n in vals], [n for _, n in vals]


surf, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
fig.patch.set_facecolor(surf); ax.set_facecolor(surf)
ax.axvspan(-0.5, 0.5, color="#f0efec", zorder=0)
notes = []
for method, name, c in MODELS:
    y, n = pooled(method)
    ax.plot(range(1, 9), y[1:], color=c, lw=2, solid_capstyle="round", zorder=2, label=name, marker="o", ms=0)
    for i, (v, k) in enumerate(zip(y, n)):
        done = k >= 100
        ax.scatter(i, v, s=64, color=c if done else surf, edgecolors=surf if done else c, linewidths=2, zorder=3)
        if not done:
            ax.annotate("*", (i, v), xytext=(5, 3), textcoords="offset points", color=ink2, fontsize=11, fontweight="bold")
            notes.append(f"{name.split(' (')[-1].rstrip(')')} {COLS[i].replace(chr(10), ' ')} n={k}")
ax.set_xticks(range(9), COLS, color=ink2, fontsize=9); ax.set_xlim(-0.5, 8.4); ax.set_ylim(-3, 103)
ax.set_ylabel("Success rate (%)", color=ink2)
ax.set_xlabel("Camera miscalibration (held-out G4, rotation° / translation cm)", color=ink2)
ax.grid(axis="y", color=grid, lw=1); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(grid)
ax.tick_params(colors=ink2)
ax.text(0, 101, "seen G1", ha="center", color=ink2, fontsize=8)
ax.set_title("Handover success vs camera miscalibration (100 seed-matched episodes per point)", color=ink, fontsize=11, loc="left")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9, labelcolor=ink)
foot = "Every model runs all 100 handover test episodes with seed 0. Dense single-task models."
if notes: foot += "  * incomplete: " + ", ".join(notes) + "."
fig.text(0.01, 0.01, foot, color=ink2, fontsize=8)
plt.subplots_adjust(right=0.72, bottom=0.17)
fig.savefig("docs/research/experiments/2026-10-20/figures/exp10_seedmatched_success_vs_miscal.png", facecolor=surf)
print("saved")
