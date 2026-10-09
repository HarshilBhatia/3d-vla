#!/usr/bin/env python3
"""Tables and figures for TODO 14b: TODO 5/11 probes through exp 4a's pre-LN pooled extractor vs exp 3, from results/analysis/probes_exp4a/fits/."""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from plot_probes import FIG, INK, MUTED, agg

RES = "results/analysis/probes_exp4a"
PANELS = [("A_group", "acc", "iid", "Camera group (G1–G6), val accuracy", "accuracy", 1 / 6),
          ("A_pos", "err_cm", "logo", "Camera position, leave-one-group-out", "mean error (cm)", 50.2),
          ("B_rotvec", "r2", "iid", "Rotation error vector (0–60° sweep), val R²", "R²", 0),
          ("B_trans", "r2", "iid", "Translation error vector (0–60 cm sweep), val R²", "R²", 0),
          ("B_clean_vs_sweep", "acc", "iid", "Clean vs miscalibrated (sweep), val accuracy", "accuracy", 0.5)]
TICKS = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
LABELS = ["enc", "ST1", "TP1", "ST2", "TP2", "ST3", "TP3", "ST4", "TP4", "ext out", "SA1", "SA2", "SA3", "SA4"]


def stages():
    """(x, C-token feature, delta_M feature whose input is that C-token or None)."""
    out = [(0.0, "enc", None)]
    for i in range(4):
        out += [(1 + i, f"x{i}st_tok", None), (1.5 + i, f"x{i}tp_tok", None)]
    out.append((5.5, "hist", "dm0@last"))
    out += [(6.5 + l, f"sa{l}@last", f"dm{l + 1}@last" if l < 3 else None) for l in range(4)]
    return out


def value(rows, model, feature, t, met, sp):
    sel = dict(model=model, feature=feature, target=t, split=sp, cam=[0, 1])
    if sp == "iid":
        sel["shuffled"] = False
    return agg(rows, **sel).get(met)


def rel(c, s):
    c, s = c[:, :2], s[:, :2]
    norm = c.mean(0).norm(dim=-1).mean()
    return float(c.std(0).norm(dim=-1).mean() / norm), float((c - s).norm(dim=-1).mean() / norm)


def scale_stats(model):
    """Per feature: across-sample std / mean norm (clean val, cams 0+1), and mean |clean - sweep| / norm on the same sample."""
    F = torch.load(f"{RES}/{model}.pt", weights_only=False)["val"]["feats"]
    out = {k: rel(c.float(), F["sweep"][k].float()) for k, c in F["clean"].items() if c.ndim == 3}
    for l in range(4):
        out[f"sa{l}@last"] = rel(F["clean"]["sa"][:, 1, l].float(), F["sweep"]["sa"][:, 1, l].float())
        d = F["clean"]["dm"].shape[2] - 4 + l
        out[f"dm{l}@last"] = rel(F["clean"]["dm"][:, 1, d].flatten(2).float(), F["sweep"]["dm"][:, 1, d].flatten(2).float())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", required=True, help="checkpoint label, e.g. 120k or 200k")
    a = p.parse_args()
    models = {f"exp4a_{a.step}": (f"exp 4a pre-LN ({a.step} EMA)", "#2a78d6"), f"exp3_{a.step}": (f"exp 3 ({a.step} EMA)", "#eb6834")}
    rows = []
    for m in models:
        for f in glob.glob(f"{RES}/fits/probe_results_{m}.json"):
            rows += json.load(open(f))
    fig, axs = plt.subplots(len(PANELS), 1, figsize=(11, 3.0 * len(PANELS)))
    for ax, (t, met, sp, title, yl, ref) in zip(axs, PANELS):
        for m, (lab, col) in models.items():
            pts = [(x, value(rows, m, f, t, met, sp)) for x, f, _ in stages()]
            pts = [(x, v) for x, v in pts if v is not None]
            ax.plot([x for x, _ in pts], [v for _, v in pts], color=col, lw=2, marker="o", ms=4, label=lab)
            pts = [(x, value(rows, m, d, t, met, sp)) for x, _, d in stages() if d]
            pts = [(x, v) for x, v in pts if v is not None]
            ax.plot([x for x, _ in pts], [v for _, v in pts], color=col, lw=1.2, ls="--", marker="s", ms=6, mfc="white")
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
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"probe_exp4a_{a.step}.{ext}", dpi=150, facecolor="#fcfcfb")
    lines = ["| Model | Stage | spread | sweep shift | group acc | LOGO pos cm | rotvec R² | trans vec R² | rot mag R² | clean/sweep acc |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m, (lab, _) in models.items():
        sc = scale_stats(m)
        for _, f, d in stages():
            for name in (f, d):
                if name is None:
                    continue
                vals = [value(rows, m, name, t, met, sp) for t, met, sp in (("A_group", "acc", "iid"), ("A_pos", "err_cm", "logo"),
                        ("B_rotvec", "r2", "iid"), ("B_trans", "r2", "iid"), ("B_rot_deg", "r2", "iid"), ("B_clean_vs_sweep", "acc", "iid"))]
                if all(v is None for v in vals):
                    continue
                sp_, sh_ = sc.get(name, (float("nan"), float("nan")))
                txt = ["–" if v is None else (f"{v:.0f}" if i == 1 else f"{v:.2f}") for i, v in enumerate(vals)]
                lines.append(f"| {lab} | {name} | {sp_:.1e} | {sh_:.1e} | " + " | ".join(txt) + " |")
    Path(f"{RES}/summary_table_{a.step}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
