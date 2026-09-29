"""SBRS checkpoint-ladder success vs total training steps (from random init) for the history ablations and Base."""
import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.plotting import week_figures_dir

R20 = Path("/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts")
OUT = week_figures_dir()
# (label, ladder campaign prefix, steps inherited before this run's step 0, colour, marker).
RUNS = [
    ("VideoPooled (+100k scratch stage)", "video_deltam_checkpoint_ladder_v01_video-pooled-k5", 100_000, "#2a78d6", "o"),
    ("Random hist, δM on (+184k from VP84k)", "video_deltam_checkpoint_ladder_v01_vid-deltaM-pooled-randhist-notime", 184_000, "#eb6834", "s"),
    ("No δM, ordered (+184k from VP84k)", "video_deltam_checkpoint_ladder_v01_vid-deltaM-pooled-no-view-align", 184_000, "#1baf7a", "^"),
    ("A: scratch, no δM, random", "video_deltam_checkpoint_ladder_v01_vid-pooled-noalign-randhist-scratch", 0, "#eda100", "D"),
    ("C: A − register (+50k from A)", "video_deltam_checkpoint_ladder_v01_vid-pooled-noalign-randhist-noreg-from-a50k", 50_000, "#e87ba4", "v"),
    ("Base (scratch)", "sbrs_checkpoint_ladder_v01_base", 0, "#8a8985", "x"),
]
TASKS = ["handover_item_easy", "lift_ball", "pick_laptop", "pick_plate", "push_box", "straighten_rope"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def ladder(prefix, offset):
    points = {}
    for d in glob.glob(str(R20 / f"{prefix}_s*")):
        step = int(d.rsplit("_s", 1)[1])
        per = {}
        # Every ladder shares the trained-bias + 3 deg / 1 cm residual condition; Base's also has e0, ignored here.
        for f in glob.glob(f"{d}/*/*/*e3deg-t1cm*/results_*.json"):
            if f.endswith((".manifest.json", ".progress.json")):
                continue
            task, res = next(iter(json.load(open(f)).items()))
            per[task.replace("bimanual_", "")] = 100 * res["mean"]
        if len(per) == len(TASKS):
            points[(step + offset) / 1000] = per
    return dict(sorted(points.items()))


def main():
    data = {label: ladder(prefix, off) for label, prefix, off, _, _ in RUNS}
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4), sharey=True, sharex=True, facecolor="#fcfcfb")
    axes = axes.ravel()
    rows = []
    for ax, panel in zip(axes, ["mean over 6 tasks"] + TASKS):
        ax.set_ylim(-3, 103)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=7)
        for label, _, _, colour, marker in RUNS:
            pts = data[label]
            xs = list(pts)
            ys = [sum(v.values()) / len(v) if panel.startswith("mean") else v[panel] for v in pts.values()]
            rows += [[label, panel, x, round(y, 1)] for x, y in zip(xs, ys)]
            base = label.startswith("Base")
            ax.plot(xs, ys, color=colour, linewidth=1.6 if base else 2, linestyle="--" if base else "-",
                    marker=marker, markersize=4, label=label, zorder=2 if base else 3)
        ax.set_title(panel, fontsize=9, color=TEXT, fontweight="bold" if panel.startswith("mean") else "normal")
    axes[-1].axis("off")
    for ax in (axes[0], axes[4]):
        ax.set_ylabel("SBRS success %", fontsize=8, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("SBRS checkpoint ladder (seen cameras, trained bias + 3°/1 cm, 6 tasks × 20 rollouts) vs training steps",
                 fontsize=11, color=TEXT, y=1.06)
    fig.supxlabel("total training steps since random init (k); warm-started runs are shifted by what they inherited",
                  fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "ladder_curves_sbrs.png", dpi=160, bbox_inches="tight")
    with open(OUT / "ladder_curves_sbrs.csv", "w", newline="") as f:
        csv.writer(f).writerows([["model", "panel", "total_steps_k", "success_pct"]] + rows)
    print("wrote", OUT / "ladder_curves_sbrs.png", {k: len(v) for k, v in data.items()})


TASK_STYLE = list(zip(TASKS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"], "os^Dv<"))


def per_run():
    """One panel per run: each checkpoint's 6 task success rates and their mean, x = that run's own step."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), sharey=True, facecolor="#fcfcfb")
    for ax, (label, prefix, offset, _, _) in zip(axes.ravel(), RUNS):
        pts = ladder(prefix, 0)
        xs = list(pts)
        ax.set_ylim(-3, 103)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=7)
        for task, colour, marker in TASK_STYLE:
            ax.plot(xs, [v[task] for v in pts.values()], color=colour, linewidth=1.3, marker=marker,
                    markersize=3.5, alpha=0.9, label=task)
        ax.plot(xs, [sum(v.values()) / len(v) for v in pts.values()], color=TEXT, linewidth=2.6, marker="o",
                markersize=4.5, label="mean over 6 tasks", zorder=4)
        name = label.split(" (")[0]
        ax.set_title(f"{name}\n(steps in this run; +{offset // 1000}k inherited)" if offset else f"{name}\n(from scratch)",
                     fontsize=8.5, color=TEXT)
        if not pts:
            ax.text(0.5, 0.5, "K=5 ladder still running", transform=ax.transAxes, ha="center", color=MUTED, fontsize=9)
            ax.set_xticks([])
        ax.set_xlabel("checkpoint step (k)", fontsize=7.5, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("SBRS success %", fontsize=8, color=MUTED)
    handles, labels = axes.ravel()[3].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("SBRS ladder per run: task success and mean at each checkpoint (6 tasks × 20 rollouts)",
                 fontsize=11, color=TEXT, y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "ladder_per_run_sbrs.png", dpi=160, bbox_inches="tight")
    print("wrote", OUT / "ladder_per_run_sbrs.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=OUT,
                    help="where to write the figures (default: this week's experiment figures dir)")
    OUT = ap.parse_args().output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    main()
    per_run()
