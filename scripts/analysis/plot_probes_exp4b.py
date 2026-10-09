#!/usr/bin/env python3
"""Tables and figures for TODO 15: task / camera-group fingerprint probes on exp 4b (input skip) vs 4a and exp 3, from results/analysis/probes_exp4b/fits/."""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_probes import FIG, INK, MUTED, agg

RES = "results/analysis/probes_exp4b"
PANELS = [("task", "acc", "iid", "Task (13-way), val accuracy", "accuracy", 1 / 13),
          ("A_group", "acc", "iid", "Camera group (G1–G6), val accuracy", "accuracy", 1 / 6),
          ("A_pos", "err_cm", "logo", "Camera position, leave-one-group-out", "mean error (cm)", 50.2),
          ("B_rotvec", "r2", "iid", "Rotation error vector (0–60° sweep), val R²", "R²", 0),
          ("B_clean_vs_sweep", "acc", "iid", "Clean vs miscalibrated (sweep), val accuracy", "accuracy", 0.5)]
TICKS = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5.0, 5.5, 6.5, 7.5, 8.5, 9.5]
LABELS = ["enc", "ST1", "TP1", "ST2", "TP2", "ST3", "TP3", "ST4", "TP4\n(stack)", "skip", "ext out", "SA1", "SA2", "SA3", "SA4"]
NAMES = {"enc": "enc", "x3tp_tok": "stack out (TP4)", "xskip": "skip term", "hist": "extractor out",
         "sa0@last": "SA1", "sa1@last": "SA2", "sa2@last": "SA3", "sa3@last": "SA4",
         "dm0@last": "delta_M (from ext out)", "dm1@last": "delta_M (from SA1)", "dm2@last": "delta_M (from SA2)", "dm3@last": "delta_M (from SA3)"}


def stages():
    """(x, C-token feature, delta_M feature whose input is that C-token or None)."""
    out = [(0.0, "enc", None)]
    for i in range(4):
        out += [(1 + i, f"x{i}st_tok", None), (1.5 + i, f"x{i}tp_tok", None)]
    out += [(5.0, "xskip", None), (5.5, "hist", "dm0@last")]
    out += [(6.5 + l, f"sa{l}@last", f"dm{l + 1}@last" if l < 3 else None) for l in range(4)]
    return out


def value(rows, model, feature, t, met, sp):
    sel = dict(model=model, feature=feature, target=t, split=sp, cam=[0, 1])
    if sp == "iid":
        sel["shuffled"] = False
    return agg(rows, **sel).get(met)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", required=True, help="checkpoint label, e.g. 160k or 200k")
    a = p.parse_args()
    models = {f"exp4b_{a.step}": (f"exp 4b input skip ({a.step})", "#1baf7a"), f"exp4a_{a.step}": (f"exp 4a pre-LN ({a.step})", "#2a78d6"),
              f"exp3_{a.step}": (f"exp 3 ({a.step})", "#eb6834")}
    rows = []
    for m in models:
        for f in glob.glob(f"{RES}/fits/probe_results_{m}.json"):
            rows += json.load(open(f))
    fig, axs = plt.subplots(len(PANELS), 1, figsize=(11, 3.0 * len(PANELS)))
    for ax, (t, met, sp, title, yl, ref) in zip(axs, PANELS):
        for m, (lab, col) in models.items():
            for idx, ls, mk in ((1, "-", "o"), (2, "--", "s")):
                pts = [(s[0], value(rows, m, s[idx], t, met, sp)) for s in stages() if s[idx]]
                pts = [(x, v) for x, v in pts if v is not None]
                if pts:
                    ax.plot([x for x, _ in pts], [v for _, v in pts], color=col, lw=2 if idx == 1 else 1.2, ls=ls, marker=mk,
                            ms=4 if idx == 1 else 6, mfc=col if idx == 1 else "white", label=lab if idx == 1 else None)
        ax.axhline(ref, color=MUTED, lw=1, ls="--")
        ax.axvspan(0.6, 4.9, color="#f0efec", zorder=0)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_ylabel(yl, fontsize=9, color=MUTED)
        ax.set_xticks(TICKS); ax.set_xticklabels(LABELS, fontsize=8, color=MUTED)
        ax.tick_params(axis="y", labelsize=8, colors=MUTED)
        ax.grid(axis="y", color="#e6e5e1", lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, l = axs[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=MUTED, ls="--", marker="s", mfc="white"))
    l.append("delta_M predicted from that C-token")
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"probe_exp4b_{a.step}.{ext}", dpi=150, facecolor="#fcfcfb")
    cols = [("task", "acc", "iid"), ("A_group", "acc", "iid"), ("A_pos", "err_cm", "logo"), ("B_rotvec", "r2", "iid"), ("B_clean_vs_sweep", "acc", "iid")]
    lines = ["| Stage | " + " | ".join(f"{lab.split(' (')[0]}: task / group / LOGO cm / rotvec R² / clean-sweep" for lab, _ in models.values()) + " |",
             "|---|" + "---|" * len(models)]
    for name, label in NAMES.items():
        cells = []
        for m in models:
            v = [value(rows, m, name, t, met, sp) for t, met, sp in cols]
            cells.append(" / ".join("–" if x is None else (f"{x:.0f}" if i == 2 else f"{x:.2f}") for i, x in enumerate(v)))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    Path(f"{RES}/summary_table_{a.step}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
