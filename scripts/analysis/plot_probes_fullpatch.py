#!/usr/bin/env python3
"""Tables and figures for TODO 11: TODO 5's probes through the full-patch history extractor, from results/analysis/probes_fullpatch/fits/."""
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

RES = "results/analysis/probes_fullpatch"
MODELS = {"refine_rope3d": ("Refine FullPatch rope3d", "#2a78d6"), "direct_rope3d": ("Direct FullPatch rope3d", "#eb6834"),
          "a6000_b64": ("FullPatch a6000_b64", "#1baf7a"), "d8": ("FullPatch d8", "#eda100"),
          "randinit_rope3d": ("random-init Refine rope3d", "#8a8984")}
PANELS = [("A_group", "acc", "iid", "Camera group (G1–G6), val accuracy", "accuracy", 1 / 6),
          ("A_pos", "err_cm", "logo", "Camera position, leave-one-group-out", "mean error (cm)", None),
          ("B_rotvec", "r2", "iid", "Rotation error vector (0–60° sweep), val R²", "R²", 0),
          ("B_trans", "r2", "iid", "Translation error vector (0–60 cm sweep), val R²", "R²", 0),
          ("B_clean_vs_sweep", "acc", "iid", "Clean vs miscalibrated (sweep), val accuracy", "accuracy", 0.5)]


def load():
    rows = []
    for f in sorted(glob.glob(f"{RES}/fits/probe_results_*.json")):
        rows += json.load(open(f))
    return rows


def stages(rows, model):
    """(x, feature, style) along the pipeline: enc, extractor blocks (patch mean / image token), extractor out, head SA."""
    feats = {r["feature"] for r in rows if r["model"] == model}
    nblk = len([f for f in feats if f.startswith("x") and f.endswith("st_pm")])
    out = [(0.0, "enc", "pm")]
    for i in range(nblk):
        for j, (blk, kinds) in enumerate((("st", ("pm",)), ("tp", ("pm", "img")))):
            x = 0.5 + (2 * i + j + 1) * 4 / (2 * nblk)
            for k in kinds:
                out.append((x, f"x{i}{blk}_{k}", k))
    out.append((5.5, "hist", "img"))
    out += [(6.5 + l, f"sa{l}@last", "pm") for l in range(4)]
    return out


def series(rows, model, target, metric, split, which):
    xs, ys = [], []
    for x, f, k in stages(rows, model):
        if which == "img" and k != "img" or which == "pm" and k == "img" and f != "hist":
            continue
        sel = dict(model=model, feature=f, target=target, split=split, cam=[0, 1])
        if split == "iid":
            sel["shuffled"] = False
        v = agg(rows, **sel).get(metric)
        if v is not None:
            xs.append(x); ys.append(v)
    return xs, ys


def scale_stats(model):
    """Per feature: across-sample std / mean norm (clean val, cams 0+1), and mean |clean - sweep| / norm on the same sample."""
    p = Path(f"{RES}/{model}.pt")
    if not p.exists():
        return {}
    F = torch.load(p, weights_only=False)["val"]["feats"]
    out = {}
    for k, c in F["clean"].items():
        if c.ndim != 3:
            if k == "sa":
                c, sw = c[:, 1].float(), F["sweep"][k][:, 1].float()
                for l in range(c.shape[1]):
                    out[f"sa{l}@last"] = rel(c[:, l], sw[:, l])
            continue
        out[k] = rel(c.float(), F["sweep"][k].float())
    if "xdm" in F["clean"]:
        out["xdm"] = rel(F["clean"]["xdm"].flatten(2).float(), F["sweep"]["xdm"].flatten(2).float())
    return out


def rel(c, s):
    c, s = c[:, :2], s[:, :2]
    norm = c.mean(0).norm(dim=-1).mean()
    return float(c.std(0).norm(dim=-1).mean() / norm), float((c - s).norm(dim=-1).mean() / norm)


def main():
    rows = load()
    fig, axs = plt.subplots(len(PANELS), 1, figsize=(11, 3.1 * len(PANELS)))
    ticks = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
    labels = ["enc", "ST1", "TP1", "ST2", "TP2", "ST3", "TP3", "ST4", "TP4", "ext out", "SA1", "SA2", "SA3", "SA4"]
    for ax, (t, met, sp, title, yl, ref) in zip(axs, PANELS):
        for m, (lab, col) in MODELS.items():
            x, y = series(rows, m, t, met, sp, "pm")
            if x:
                ax.plot(x, y, color=col, lw=2, marker="o", ms=4, label=lab)
            x, y = series(rows, m, t, met, sp, "img")
            if x:
                ax.plot(x, y, color=col, lw=1.2, ls="--", marker="D", ms=4, mfc="white")
            v = agg(rows, model=m, feature="xdm", target=t, split=sp, cam=[0, 1], **({"shuffled": False} if sp == "iid" else {})).get(met)
            if v is not None:
                ax.plot([5.5], [v], color=col, marker="s", ms=8, mfc="white", mew=1.5, ls="none")
        if ref is not None:
            ax.axhline(ref, color=MUTED, lw=1, ls="--")
        if t == "A_pos":
            for key, ls in (("mean_baseline_cm", "--"), ("nearest_other_group_cm", "-.")):
                v = agg(rows, target="A_pos", split="logo", feature="enc", cam=[0, 1]).get(key)
                if v:
                    ax.axhline(v, color=MUTED, lw=1, ls=ls)
        ax.axvspan(0.6, 4.9, color="#f0efec", zorder=0)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_ylabel(yl, fontsize=9, color=MUTED)
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8, color=MUTED)
        ax.tick_params(axis="y", labelsize=8, colors=MUTED)
        ax.grid(axis="y", color="#e6e5e1", lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, l = axs[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color=MUTED, ls="--", marker="D", mfc="white"), plt.Line2D([], [], color=MUTED, marker="s", mfc="white", ls="none")]
    l += ["per-image token (solid = patch mean)", "Direct delta_M (extractor output)"]
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"probe_fullpatch_extractor.{ext}", dpi=150, facecolor="#fcfcfb")
    lines = ["| Model | Stage | spread | sweep shift | group acc | LOGO pos cm | rotvec R² | trans vec R² | rot mag R² | clean/sweep acc |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m, (lab, _) in MODELS.items():
        sc = scale_stats(m)
        for x, f, k in stages(rows, m) + [(5.5, "xdm", "dm")]:
            vals = []
            for t, met, sp in (("A_group", "acc", "iid"), ("A_pos", "err_cm", "logo"), ("B_rotvec", "r2", "iid"), ("B_trans", "r2", "iid"),
                               ("B_rot_deg", "r2", "iid"), ("B_clean_vs_sweep", "acc", "iid")):
                sel = dict(model=m, feature=f, target=t, split=sp, cam=[0, 1])
                if sp == "iid":
                    sel["shuffled"] = False
                v = agg(rows, **sel).get(met)
                vals.append("–" if v is None else (f"{v:.0f}" if met == "err_cm" else f"{v:.2f}"))
            if any(v != "–" for v in vals):
                sp_, sh_ = sc.get(f, (None, None))
                pre = "– | –" if sp_ is None else f"{sp_:.1e} | {sh_:.1e}"
                lines.append(f"| {lab} | {f} | {pre} | " + " | ".join(vals) + " |")
    Path(f"{RES}/summary_table.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
