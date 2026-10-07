#!/usr/bin/env python3
"""Tables and figures for the TODO 5 linear probes, from results/analysis/probes/probe_results_*.json."""
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = "results/analysis/probes"
FIG = Path("docs/research/experiments/2026-10-06/figures")
MODELS = {"exp3": ("exp 3 (200k EMA)", "#2a78d6"), "1b": ("1b (200k EMA)", "#eb6834"),
          "controlA": ("control A (200k raw)", "#1baf7a"), "ema20k_randhist": ("ema20k randhist (15k EMA)", "#eda100"),
          "randinit_exp3": ("random-init exp 3 arch", "#8a8984")}
DEPTHS = ["enc", "hist", "SA1", "SA2", "SA3", "SA4"]
INK, MUTED = "#0b0b0b", "#52514e"


def load():
    rows = []
    for f in sorted(glob.glob(f"{RES}/fits/probe_results_*.json")):
        rows += json.load(open(f))
    return rows


def agg(rows, **sel):
    """Mean of each metric over the rows matching sel (typically cameras 0 and 1)."""
    out = defaultdict(list)
    for r in rows:
        if all((r.get(k) in v) if isinstance(v, (list, tuple, set)) else r.get(k) == v for k, v in sel.items()):
            for k, v in r.items():
                if isinstance(v, float):
                    out[k].append(v)
    return {k: float(np.mean(v)) for k, v in out.items()}


def ctok_name(d, step):
    return ["enc", "hist"][d] if d < 2 else f"sa{d - 2}@{step}"


def curve(rows, model, target, metric, split="iid", step="last", cams=(0, 1), kind="ctok"):
    xs, ys = [], []
    for d in range(6):
        if kind == "ctok":
            name = ctok_name(d, step)
        elif kind == "dm":
            if not 1 <= d <= 4:
                continue
            name = f"dm{d - 1}@{step}"
        else:
            if not 1 <= d <= 5:
                continue
            name = f"ad{d - 1}@{step}"
        sel = dict(model=model, target=target, split=split, feature=name, cam=list(cams) if kind != "ad" else "global")
        if split == "iid":
            sel["shuffled"] = False
        a = agg(rows, **sel)
        if metric in a:
            xs.append(d); ys.append(a[metric])
    return xs, ys


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.set_xticks(range(6)); ax.set_xticklabels(DEPTHS, fontsize=8, color=MUTED)
    ax.tick_params(axis="y", labelsize=8, colors=MUTED)
    ax.grid(axis="y", color="#e6e5e1", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")


def panel(ax, rows, target, metric, split="iid", mech=True):
    for m, (lab, col) in MODELS.items():
        x, y = curve(rows, m, target, metric, split)
        if x:
            ax.plot(x, y, color=col, lw=2, marker="o", ms=5, label=lab)
        x, y = curve(rows, m, target, metric, split, step="first")
        if x:
            ax.plot(x[2:], y[2:], color=col, lw=1, ls=":", alpha=0.8)
        if mech:
            for kind, mk in (("dm", "s"), ("ad", "D")):
                x, y = curve(rows, m, target, metric, split, kind=kind)
                if x:
                    ax.plot(x, y, color=col, lw=1, ls="--", marker=mk, ms=6, mfc="white", mew=1.5)


def fig_pose(rows):
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
    panel(axs[0, 0], rows, "A_group", "acc")
    axs[0, 0].axhline(1 / 6, color=MUTED, lw=1, ls="--"); axs[0, 0].text(5, 1 / 6 + 0.02, "chance 1/6", fontsize=7, color=MUTED, ha="right")
    tb = agg(rows, feature="task_onehot", target="A_group").get("acc")
    if tb:
        axs[0, 0].axhline(tb, color=MUTED, lw=1, ls="-."); axs[0, 0].text(5, tb + 0.02, f"task one-hot {tb:.2f}", fontsize=7, color=MUTED, ha="right")
    style(axs[0, 0], "Camera group (G1–G6), val accuracy", "accuracy")
    panel(axs[0, 1], rows, "A_pos", "err_cm")
    style(axs[0, 1], "Camera position, held-out episodes (6 poses seen)", "mean error (cm)")
    panel(axs[1, 0], rows, "A_pos", "err_cm", split="logo")
    for key, ls, txt in (("mean_baseline_cm", "--", "predict mean of 5 train poses"), ("nearest_other_group_cm", "-.", "nearest other pose")):
        v = agg(rows, model="exp3", target="A_pos", split="logo", feature="enc", cam=[0, 1])[key]
        axs[1, 0].axhline(v, color=MUTED, lw=1, ls=ls); axs[1, 0].text(5, v + 1, f"{txt} {v:.0f}", fontsize=7, color=MUTED, ha="right")
    style(axs[1, 0], "Camera position, leave-one-group-out", "mean error (cm)")
    panel(axs[1, 1], rows, "A_fwd", "err_deg", split="logo")
    for key, ls, txt in (("mean_baseline_deg", "--", "predict mean of 5 train poses"), ("nearest_other_group_deg", "-.", "nearest other pose")):
        v = agg(rows, model="exp3", target="A_fwd", split="logo", feature="enc", cam=[0, 1])[key]
        axs[1, 1].axhline(v, color=MUTED, lw=1, ls=ls); axs[1, 1].text(5, v + 0.5, f"{txt} {v:.0f}", fontsize=7, color=MUTED, ha="right")
    style(axs[1, 1], "Viewing direction, leave-one-group-out", "mean angle error (deg)")
    finish(fig, axs, "probe_pose")


def fig_miscal(rows):
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (t, met, title, yl) in zip(axs.flat, [
            ("B_rot_deg", "r2", "Rotation error magnitude (0–60°), val R²", "R²"),
            ("B_rotvec", "r2", "Rotation error vector (world axis-angle), val R²", "R²"),
            ("B_trans", "r2", "Translation error vector (world), val R²", "R²"),
            ("B_clean_vs_sweep", "acc", "Clean vs miscalibrated (sweep), val accuracy", "accuracy")]):
        panel(ax, rows, t, met)
        ax.axhline(0.5 if met == "acc" else 0, color=MUTED, lw=1, ls="--")
        style(ax, title, yl)
    finish(fig, axs, "probe_miscal")


def finish(fig, axs, name):
    h, l = axs[0, 0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color=MUTED, lw=1, ls=":"), plt.Line2D([], [], color=MUTED, ls="--", marker="s", mfc="white"),
          plt.Line2D([], [], color=MUTED, ls="--", marker="D", mfc="white")]
    l += ["C-tokens, first flow step (solid = last)", "delta_M per camera (at its input depth)", "1b adaLN camera term (global)"]
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def table(rows):
    """Markdown tables (cams 0+1 averaged, C-tokens at the last flow step)."""
    lines = []
    feats = [("enc", "enc"), ("hist", "hist")] + [(f"SA{l + 1}", f"sa{l}@last") for l in range(4)]
    cols = [("group acc", "A_group", "acc", "iid"), ("pos err cm", "A_pos", "err_cm", "iid"),
            ("LOGO pos cm", "A_pos", "err_cm", "logo"), ("LOGO pos R²", "A_pos", "r2", "logo"), ("LOGO fwd °", "A_fwd", "err_deg", "logo"),
            ("rot mag R²", "B_rot_deg", "r2", "iid"), ("rot mag MAE °", "B_rot_deg", "mae", "iid"),
            ("trans mag R²", "B_trans_cm", "r2", "iid"), ("trans mag MAE cm", "B_trans_cm", "mae", "iid"),
            ("rotvec R²", "B_rotvec", "r2", "iid"), ("trans vec R²", "B_trans", "r2", "iid"),
            ("clean/sweep acc", "B_clean_vs_sweep", "acc", "iid"), ("clean/train acc", "B_clean_vs_train", "acc", "iid")]
    hdr = "| Model | Feature | " + " | ".join(c[0] for c in cols) + " |"
    lines += [hdr, "|" + "---|" * (len(cols) + 2)]

    def row(m, label, name, cam):
        vals = []
        for _, t, met, sp in cols:
            sel = dict(model=m, feature=name, target=t, split=sp, cam=cam)
            if sp == "iid":
                sel["shuffled"] = False
            v = agg(rows, **sel).get(met)
            vals.append("–" if v is None else (f"{v:.2f}" if met in ("acc", "r2") else f"{v:.1f}"))
        lines.append(f"| {MODELS[m][0]} | {label} | " + " | ".join(vals) + " |")

    for m in MODELS:
        for label, name in feats:
            row(m, label, name, [0, 1])
        for l in range(4):
            if agg(rows, model=m, feature=f"dm{l}@last"):
                row(m, f"delta_M (SA{l + 1})", f"dm{l}@last", [0, 1])
        for k in range(5):
            if agg(rows, model=m, feature=f"ad{k}@last"):
                row(m, f"adaLN term ({'hist' if k == 0 else f'after SA{k}'})", f"ad{k}@last", "global")
        row(m, "wrist-mean C-token enc", "enc", ["wrist->0", "wrist->1"])
        row(m, "wrist-mean C-token SA4", "sa3@last", ["wrist->0", "wrist->1"])
    lines.append("")
    lines.append("Shuffled-label controls (max over models and features): " + ", ".join(
        f"{t} {met} {max(r[met] for r in rows if r.get('shuffled') and r['target'] == t and met in r):.2f}"
        for t, met in (("A_group", "acc"), ("A_pos", "r2"), ("B_rot_deg", "r2"), ("B_trans_cm", "r2"))))
    w = []
    for m in MODELS:
        a = [agg(rows, model=m, feature=n, target=f"A_{k}", split="iid", shuffled=False, cam=[2, 3]).get(met)
             for n in ("enc", "sa3@last") for k, met in (("pos", "r2"), ("pos", "err_cm"), ("fwd", "err_deg"))]
        if a[0] is not None:
            w.append(f"| {MODELS[m][0]} | " + " | ".join(f"{x:.2f}" if i % 3 == 0 else f"{x:.1f}" for i, x in enumerate(a)) + " |")
    lines += ["", "Wrist cameras' own pose (continuous), cams 2+3:", "",
              "| Model | enc pos R² | enc pos err cm | enc fwd err ° | SA4 pos R² | SA4 pos err cm | SA4 fwd err ° |", "|---|---|---|---|---|---|---|"] + w
    return "\n".join(lines)


def main():
    rows = load()
    fig_pose(rows)
    fig_miscal(rows)
    t = table(rows)
    Path(f"{RES}/summary_table.md").write_text(t)
    print(t)


if __name__ == "__main__":
    main()
