#!/usr/bin/env python3
"""Fraction of the clean-to-heavy60 degradation recovered (or caused) by each patched component, from activation_patching.py recordings (TODO 7)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.plotting import configure_theme, save_figure

MODELS = {  # label -> (legend name, categorical slot colour), fixed order shared with TODO 6 figures
    "exp3_200k": ("exp 3, 200k (EMA)", "#2a78d6"),
    "adaln1b_200k": ("1b, 200k (EMA)", "#1baf7a"),
    "controlA_200k": ("control A, 200k (raw)", "#eda100"),
    "deltam140k": ("DeltaM 140k, no history (raw)", "#eb6834"),
    "base160k": ("Base 160k, no history (raw)", "#e87ba4"),
    "fp": ("FP (full-patch, a6000_b64)", "#008300"),
    "fp_d8": ("FP-d8", "#4a3aa7"),
    "fp_r3d_refine": ("FP-R3D-refine", "#e34948"),
    "fp_r3d_direct": ("FP-R3D-direct", "#606B75"),
    "exp3_120000": ("exp 3, 120k (EMA)", "#2a78d6"),
    "exp4a_120000": ("exp 4a (pre-LN extractor), 120k (EMA)", "#eb6834"),
    "exp3_200000": ("exp 3, 200k (EMA)", "#2a78d6"),
    "exp4a_200000": ("exp 4a (pre-LN extractor), 200k (EMA)", "#eb6834"),
}
METRICS = {"pos_l2": ("position L2", 100.0, "cm"), "pos_acc_001": ("acc < 1 cm", 100.0, "%"),
           "rot_l1": ("rotation L1", 1.0, "")}
FWD = ["pos_dense", "pos_fps", "pos_ctok", "pos_dense+pos_fps", "pos_all", "proprio_feats", "pos_ctok+proprio",
       "pos_all+proprio", "mech", "pos_all+mech", "proprio+mech", "pos_ctok+mech", "ext_xyz", "ext_out",
       "pos_all+ext_xyz", "ext_xyz+proprio", "pos_all+ext_xyz+proprio", "ctok_feats+proprio", "direct_dm",
       "pos_all+direct_dm", "proprio+direct_dm", "feats_scene", "ctok_feats", "lang_query"]
REV = ["pos_dense", "pos_fps", "pos_ctok", "pos_dense+pos_fps", "pos_all", "proprio_feats", "pos_ctok+proprio",
       "pos_all+proprio", "mech", "ext_xyz", "pos_all+ext_xyz", "ext_xyz+proprio", "pos_all+ext_xyz+proprio",
       "direct_dm"]
FIG = Path("docs/research/experiments/2026-10-06/figures")
RNG = np.random.default_rng(0)


def frac(c, h, p, rev, idx):
    gap = h[idx].mean() - c[idx].mean()
    num = (p[idx].mean() - c[idx].mean()) if rev else (h[idx].mean() - p[idx].mean())
    return num / gap


def stat(c, h, p, rev, sub=None, n=1000):
    """Fraction recovered (fwd) or caused (rev) with a paired 95% bootstrap CI over samples."""
    base = np.arange(len(c)) if sub is None else np.where(sub)[0]
    est = frac(c, h, p, rev, base)
    bs = [frac(c, h, p, rev, RNG.choice(base, len(base))) for _ in range(n)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def analyse(rec):
    m = {k: {q: v.double().numpy() for q, v in d.items()} for k, d in rec["metrics"].items()}
    c, h = m["clean"], m["heavy"]
    damaged = (h["pos_l2"] - c["pos_l2"]) > 0.01
    out = {"mechanism": rec["mechanism"], "exactness": rec["exactness"], "n": int(len(c["pos_l2"])),
           "n_damaged_1cm": int(damaged.sum()),
           "input_diffs": {k: v for k, v in rec["input_diffs"].items()},
           "clean": {q: float(v.mean()) for q, v in c.items()}, "heavy": {q: float(v.mean()) for q, v in h.items()},
           "clean_damaged": {q: float(v[damaged].mean()) for q, v in c.items()},
           "heavy_damaged": {q: float(v[damaged].mean()) for q, v in h.items()}, "fwd": {}, "rev": {}}
    for d, names in (("fwd", FWD), ("rev", REV)):
        for name in names:
            key = f"{d}:{name}"
            if key not in m:
                continue
            p = m[key]
            out[d][name] = {q: {"patched": float(p[q].mean()), "frac": stat(c[q], h[q], p[q], d == "rev"),
                                "frac_damaged": stat(c[q], h[q], p[q], d == "rev", damaged)} for q in METRICS}
    return out


def fig(S, stem, title):
    configure_theme()
    nf = sum(any(n in s["fwd"] for s in S.values()) for n in FWD)
    fig, axes = plt.subplots(4, 1, figsize=(max(12, 0.32 * len(S) * nf), 15))
    for row, q in enumerate(("pos_l2", "rot_l1")):
        for col, (d, names) in enumerate((("fwd", FWD), ("rev", REV))):
            names = [n for n in names if any(n in s[d] for s in S.values())]
            ax = axes[2 * row + col]
            w = 0.8 / len(S)
            for k, (label, s) in enumerate(S.items()):
                name, colr = MODELS[label]
                x, v = [], []
                for i, n in enumerate(names):
                    if n in s[d]:
                        x.append(i + (k - (len(S) - 1) / 2) * w)
                        v.append(s[d][n][q]["frac"])
                v = np.array(v)
                ax.bar(x, v[:, 0], w * 0.92, color=colr, label=name,
                       yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], error_kw={"lw": 0.8, "capsize": 1.5})
            ax.axhline(0, color="#4E5965", lw=0.8)
            ax.axhline(1, color="#4E5965", lw=0.8, ls=":")
            ax.set_xticks(range(len(names)), [n.replace("+", "+\n") for n in names], fontsize=8, rotation=0)
            ax.grid(axis="y", color="#D9DEE3", lw=0.8)
            ax.set_ylim(-0.3, 1.3)
            ax.set_title(f"{METRICS[q][0]}: " + ("clean → heavy, fraction of degradation recovered" if d == "fwd"
                                                 else "heavy → clean, fraction of degradation caused"), fontsize=10)
    axes[0].legend(frameon=False, fontsize=8.5, ncol=len(S))
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, FIG / stem)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="results/analysis/activation_patching")
    p.add_argument("--fig-stem", default="patch_fraction_recovered")
    a = p.parse_args()
    S = {l: analyse(torch.load(Path(a.in_dir) / f"{l}.pt", weights_only=False)) for l in MODELS
         if (Path(a.in_dir) / f"{l}.pt").exists()}
    Path(a.in_dir, "summary.json").write_text(json.dumps(S, indent=1))
    fig(S, a.fig_stem, "Activation patching, 853 val samples, heavy60 on cameras 0+1 (95% paired bootstrap CI; axis clipped to [−0.3, 1.3])")
    for label, s in S.items():
        print(label, s["mechanism"], s["exactness"], "n", s["n"], "damaged>1cm", s["n_damaged_1cm"])
        print("  clean", {q: round(v, 4) for q, v in s["clean"].items()}, "heavy", {q: round(v, 4) for q, v in s["heavy"].items()})
        print("  damaged subset clean", {q: round(v, 4) for q, v in s["clean_damaged"].items()},
              "heavy", {q: round(v, 4) for q, v in s["heavy_damaged"].items()})
        for d in ("fwd", "rev"):
            for n, r in s[d].items():
                print(f"  {d}:{n:16s}", " | ".join(
                    f"{q} {r[q]['patched']:.4f} f={r[q]['frac'][0]:+.2f} [{r[q]['frac'][1]:+.2f},{r[q]['frac'][2]:+.2f}]"
                    f" dmg={r[q]['frac_damaged'][0]:+.2f}" for q in METRICS))


if __name__ == "__main__":
    main()
