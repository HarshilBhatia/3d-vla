#!/usr/bin/env python3
"""Exp 10 handover success vs camera miscalibration: 8d, 10a, 10b (50k) and 10f (100k, interim while sharded evals finish)."""
import re, subprocess, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
P = "/home/harshilb/miniconda3/envs/3dfa/bin/python"
out = subprocess.run([P, "scripts/analysis/aggregate_sharded_eval.py"], capture_output=True, text=True).stdout.splitlines()[1:]
f10, fn = [], []
for line in out:
    m = re.search(r"=(\d+)/(\d+)", line); p = re.search(r"=(\d+)% \(25/25", line)
    if p: f10.append(int(p.group(1))); fn.append(25)
    else: s, n = int(m.group(1)), int(m.group(2)); f10.append(100 * s / n if n else None); fn.append(n)
cols = ["G1\ncal", "G4\ncal", "10/10", "15/15", "20/20", "30/30", "40/40", "50/50", "60/60"]
S = {"8d (base)": ([96, 91, 94, 85, 84, 72, 48, 7, 1], [25] * 9, "#2a78d6"),
     "10a (δM)": ([16, 24, 0, 4, 12, 12, 8, 4, 0], [25] * 9, "#eb6834"),
     "10b (full-patch hist + δM)": ([24, 40, 28, 28, 24, 4, 0, 0, 0], [25] * 9, "#1baf7a"),
     "10f (pooled hist + δM)": (f10, fn, "#eda100")}
surf, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150); fig.patch.set_facecolor(surf); ax.set_facecolor(surf)
x = list(range(9))
ax.axvspan(-0.5, 0.5, color="#f0efec", zorder=0)
for name, (y, n, c) in S.items():
    xs = [i for i in range(1, 9) if y[i] is not None]
    ax.plot(xs, [y[i] for i in xs], color=c, lw=2, solid_capstyle="round", zorder=2)
    for i in x:
        if y[i] is None: continue
        done = n[i] >= 25
        ax.scatter(i, y[i], s=64, color=c if done else surf, edgecolors=c if not done else surf, linewidths=2, zorder=3)
        if not done: ax.annotate("*", (i, y[i]), xytext=(5, 3), textcoords="offset points", color=ink2, fontsize=11, fontweight="bold")
    ax.plot([], [], color=c, lw=2, marker="o", ms=7, label=name)
lab = {k: v for k, v in zip(S, [8, -8, 0, 0])}
ax.set_xticks(x, cols, color=ink2, fontsize=9); ax.set_xlim(-0.5, 8.4); ax.set_ylim(-3, 103)
ax.set_ylabel("Success rate (%)", color=ink2); ax.set_xlabel("Camera miscalibration (held-out G4, rotation° / translation cm)", color=ink2)
ax.grid(axis="y", color=grid, lw=1); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(grid)
ax.tick_params(colors=ink2)
ax.text(0, 101, "seen G1", ha="center", color=ink2, fontsize=8)
ax.set_title("Handover success vs camera miscalibration: 8d, 10a, 10b (50k) and 10f (100k)", color=ink, fontsize=11, loc="left")
fig.text(0.01, 0.01, "Filled marker = complete; hollow marker + * = 10f still evaluating. 10f pools the original run and 4 shards (repeat runs of a scene all count), n per cell: " + ", ".join(str(v) for v in fn) + ").", color=ink2, fontsize=8)
leg = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9, labelcolor=ink)
plt.subplots_adjust(right=0.74, bottom=0.17)
fig.savefig("docs/research/experiments/2026-10-20/figures/exp10_success_vs_miscal.png", facecolor=surf)
print(fn, [round(v) if v is not None else None for v in f10])
