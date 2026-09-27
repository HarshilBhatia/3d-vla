"""Per-task SBRS success over the 20k EMA fine-tunes: raw vs EMA weights per run, step 0 = the starting checkpoint."""
import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R20 = Path("/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts")
OUT = Path("docs/figures")
V, B = "video_deltam_checkpoint_ladder", "sbrs_checkpoint_ladder"
# (run key, label, ladder family, the starting checkpoint's own ladder dir or None).
RUNS = [
    ("vidpooled", "VideoPooled (from best.pth 84k)", V, None),
    ("randhist", "Random hist, δM on (from 50k)", V, f"{V}_v01_vid-deltaM-pooled-randhist-notime_s050000"),
    ("noviewalign", "No δM, ordered (from 140k)", V, f"{V}_v01_vid-deltaM-pooled-no-view-align_s140000"),
    ("regtest_a", "A: scratch, register on (from 160k)", V, f"{V}_v01_vid-pooled-noalign-randhist-scratch_s160000"),
    ("regtest_c", "C: register off (from 100k)", V, f"{V}_v01_vid-pooled-noalign-randhist-noreg-from-a50k_s100000"),
    ("base", "Base (from 160k)", B, f"{B}_v01_base_s160000"),
]
TASKS = ["handover_item_easy", "lift_ball", "pick_laptop", "pick_plate", "push_box", "straighten_rope"]
TASK_STYLE = list(zip(TASKS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"], "os^Dv<"))
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def cell(d):
    per = {}
    # Every ladder shares the trained-bias + 3 deg / 1 cm condition; Base's also has e0, ignored here.
    for f in glob.glob(str(R20 / d / "*" / "*" / "*e3deg-t1cm*" / "results_*.json")):
        if f.endswith((".manifest.json", ".progress.json")):
            continue
        task, res = next(iter(json.load(open(f)).items()))
        per[task.replace("bimanual_", "")] = 100 * res["mean"]
    return per if len(per) == len(TASKS) else None


def series(run, family, start, weights):
    pts = {}
    if start and cell(start):
        pts[0] = cell(start)
    for step in (5, 10, 15, 20):
        c = cell(f"{family}_ema20k_{weights}_v01_ema20k-{run}-{weights}_s{step * 1000:06d}")
        if c:
            pts[step] = c
    return pts


def main():
    fig, axes = plt.subplots(len(RUNS), 2, figsize=(11, 2.35 * len(RUNS)), sharex=True, sharey=True, facecolor="#fcfcfb")
    rows = []
    for r, (run, label, family, start) in enumerate(RUNS):
        for c, weights in enumerate(("raw", "ema")):
            ax = axes[r, c]
            pts = series(run, family, start, weights)
            xs = list(pts)
            ax.set_ylim(-3, 103)
            ax.set_xticks([0, 5, 10, 15, 20])
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=7)
            for task, colour, marker in TASK_STYLE:
                ax.plot(xs, [v[task] for v in pts.values()], color=colour, linewidth=1.3, marker=marker,
                        markersize=3.5, alpha=0.9, label=task)
            mean = [sum(v.values()) / len(v) for v in pts.values()]
            ax.plot(xs, mean, color=TEXT, linewidth=2.6, marker="o", markersize=4.5, label="mean over 6 tasks", zorder=4)
            for x, m in zip(xs, mean):
                ax.annotate(f"{m:.0f}", (x, m), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7, color=TEXT)
            rows += [[run, weights, x, t, round(v[t], 1)] for x, v in pts.items() for t in TASKS]
            ax.set_title(f"{label} — {'EMA' if weights == 'ema' else 'raw'} weights", fontsize=8.5, color=TEXT)
            if c == 0:
                ax.set_ylabel("SBRS %", fontsize=8, color=MUTED)
    for ax in axes[-1]:
        ax.set_xlabel("fine-tune step (k); 0 = starting checkpoint (raw)", fontsize=7.5, color=MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("20k EMA fine-tunes: per-task SBRS success, raw vs EMA weights (6 tasks × 20 rollouts)",
                 fontsize=11, color=TEXT, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(OUT / "ema20k_ladders_per_task.png", dpi=160, bbox_inches="tight")
    with open(OUT / "ema20k_ladders_per_task.csv", "w", newline="") as f:
        csv.writer(f).writerows([["run", "weights", "step_k", "task", "success_pct"]] + rows)
    print("wrote", OUT / "ema20k_ladders_per_task.png")


if __name__ == "__main__":
    main()
