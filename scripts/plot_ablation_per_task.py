"""Per-task success vs miscalibration (IDLT task-heldout) and vs G7 extremes for the five history ablations plus Base."""
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R20 = Path("/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts")
R100 = Path("/grogu/datasets/hbhatia/3dfa_online_eval_100rollouts")
OUT = Path("docs/figures")
# (label, method ids across campaigns, colour, marker); colours are the validated categorical slots 1-5.
MODELS = [
    ("VideoPooled 84k", ["video_deltam_best_s084k", "video_pooled_s084k"], "#2a78d6", "o"),
    ("Random hist 50k (δM on)", ["video_pooled_randhist_notime_s050k"], "#eb6834", "s"),
    ("No δM 140k (ordered)", ["video_pooled_no_view_align_s140k"], "#1baf7a", "^"),
    ("A: scratch, no δM, rand", ["regtest_a_register_s160k"], "#eda100", "D"),
    ("C: A − register", ["regtest_c_noregister_s100k"], "#e87ba4", "v"),
    ("Base 160k", ["base_s160k"], "#8a8985", "x"),
]
IDLT_CAMPAIGNS = ["idlt_task_heldout_group_fixed_v03_video_k5", "idlt_task_heldout_group_fixed_v02",
                  "idlt_extreme_task_heldout_group_v01", "idlt_scaled_extreme_task_heldout_group_v01",
                  "idlt_scaled_extreme_task_heldout_group_v02_5060"]
IDLT_LEVELS = [("calibrated", "0"), ("known", "bias"), ("e3deg-t1cm", "3/1"), ("e5deg-t5cm", "5/5"),
               ("e10deg-t10cm", "10"), ("e15deg-t15cm", "15"), ("e20deg-t20cm", "20"), ("e30deg-t30cm", "30"),
               ("e40deg-t40cm", "40"), ("e50deg-t50cm", "50"), ("e60deg-t60cm", "60")]
G7_TRANS = [("clean", "0"), ("opposite-100cm", "100"), ("opposite-200cm", "200"), ("opposite-500cm", "500")]
G7_ROT = [("clean", "0"), ("rot-20deg", "20°"), ("rot-90deg", "90°"), ("rot-180deg", "180°"), ("rot-270deg", "270°")]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def condition(name):
    for pat in (r"e\d+deg-t\d+cm", r"opposite-\d+cm", r"rot-\d+deg", r"calibrated", r"clean", r"known"):
        m = re.search(pat, name)
        if m:
            return m.group(0)
    return None


def load(root, campaigns):
    cells = defaultdict(dict)  # (method, condition) -> {task: success %}
    for camp in campaigns:
        for f in glob.glob(str(root / camp / "*" / "*" / "*" / "results_*.json")):
            if f.endswith((".manifest.json", ".progress.json")):
                continue
            parts = f.split("/")
            # The v02 IDLT campaign's video arm used one-frame history (the regression); only its Base cells are valid.
            if camp.endswith("fixed_v02") and parts[-4] != "base_s160k":
                continue
            task, res = next(iter(json.load(open(f)).items()))
            cells[(parts[-4], condition(parts[-2]))][task.replace("bimanual_", "")] = 100 * res["mean"]
    return cells


def series(cells, ids, level):
    for mid in ids:
        if (mid, level) in cells:
            return cells[(mid, level)]
    return {}


def style(ax):
    ax.set_ylim(-3, 103)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)


def draw(cells, levels, tasks, title, xlabel, fname, ncols, rows_out, suite):
    n = len(tasks) + 1
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows), sharey=True, facecolor="#fcfcfb")
    axes = axes.ravel()
    xs = list(range(len(levels)))
    for ax, task in zip(axes, ["mean over tasks"] + tasks):
        style(ax)
        for label, ids, colour, marker in MODELS:
            ys = []
            for lvl, _ in levels:
                v = series(cells, ids, lvl)
                if task == "mean over tasks":
                    ys.append(sum(v.values()) / len(v) if len(v) == len(tasks) else float("nan"))
                else:
                    ys.append(v.get(task, float("nan")))
                rows_out.append([suite, label, task, lvl, "" if ys[-1] != ys[-1] else round(ys[-1], 1)])
            base = label.startswith("Base")
            ax.plot(xs, ys, color=colour, linewidth=1.6 if base else 2, linestyle="--" if base else "-",
                    marker=marker, markersize=4.5, label=label, zorder=2 if base else 3)
        ax.set_title(task, fontsize=8.5, color=TEXT, fontweight="bold" if task == "mean over tasks" else "normal")
        ax.set_xticks(xs, [t for _, t in levels], fontsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    for ax in axes[::ncols]:
        ax.set_ylabel("success %", fontsize=8, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(MODELS), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, fontsize=11, color=TEXT, y=1.035)
    fig.supxlabel(xlabel, fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT / fname, dpi=160, bbox_inches="tight")
    print("wrote", OUT / fname)


def main():
    rows = []
    idlt = load(R100, IDLT_CAMPAIGNS)
    tasks13 = sorted({t for d in idlt.values() for t in d})
    draw(idlt, IDLT_LEVELS, tasks13, "IDLT task-heldout: success vs external miscalibration (13 tasks × 100 rollouts)",
         "miscalibration (rotation° / translation cm; 'bias' = trained per-group bias)", "ablation_idlt_per_task.png",
         5, rows, "idlt_heldout")
    g7 = load(R20, ["g7_extreme_opposite_v01", "g7_rotation_only_v01", "g7_opposite_camera_sbrs_v01"])
    tasks6 = sorted({t for (m, c), d in g7.items() if c == "opposite-200cm" for t in d})
    draw(g7, G7_TRANS, tasks6, "G7 extremes: opposing orbital-camera translation (6 SBRS tasks × 20 rollouts)",
         "opposing translation of the two orbital cameras (cm); 0 = calibrated G7", "ablation_g7_translation_per_task.png",
         4, rows, "g7_translation")
    draw(g7, G7_ROT, tasks6, "G7 extremes: orbital-camera rotation about world Z (6 SBRS tasks × 20 rollouts)",
         "rotation of the two orbital cameras; 0 = calibrated G7", "ablation_g7_rotation_per_task.png",
         4, rows, "g7_rotation")
    with open(OUT / "ablation_per_task.csv", "w", newline="") as f:
        csv.writer(f).writerows([["suite", "model", "task", "condition", "success_pct"]] + rows)
    print("wrote", OUT / "ablation_per_task.csv")


if __name__ == "__main__":
    main()
