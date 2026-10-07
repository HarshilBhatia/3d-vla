#!/usr/bin/env python3
"""Summarise camera_attention.py outputs: attention share per camera group, clean vs heavy60 deltas, entropy, and figures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

LAYERS = ["ca0", "ca1", "sa0", "sa1", "sa2", "sa3", "pos0", "pos1", "rot0", "rot1"]
MODELS = [("exp3_200k", "exp 3 (delta_M) 200k"), ("adaln1b_200k", "1b (adaLN) 200k"),
          ("controlA_200k", "control A 200k"), ("ema20k_randhist_15k", "warm randhist + EMA 15k")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def per_sample(d, layer):
    """Per-sample metrics averaged over steps, heads and trajectory queries."""
    m = d["mass"][layer].double()
    c = d["counts"][layer].double()
    g = m.mean(dim=(1, 2, 3))
    scene = g[:, :4].sum(-1)
    ext = g[:, :2].sum(-1)
    k = c.sum(-1)
    out = {"ext_abs": ext, "wrist_abs": g[:, 2:4].sum(-1), "ext_frac_scene": ext / scene,
           "ctok_ext": g[:, 4:6].sum(-1), "ctok_wrist": g[:, 6:8].sum(-1), "registers": g[:, 8],
           "camtoken": g[:, 9], "traj": g[:, 10],
           "ext_token_frac": c[:, :2].sum(-1) / c[:, :4].sum(-1),
           "entropy": d["ent"][layer].double().mean(dim=(1, 2, 3)),
           "entropy_norm": d["ent"][layer].double().mean(dim=(1, 2, 3)) / k.log()}
    out["ext_frac_by_step"] = (m[..., :2].sum(-1) / m[..., :4].sum(-1)).mean(dim=(2, 3))
    out["ext_frac_by_head"] = (m[..., :2].sum(-1) / m[..., :4].sum(-1)).mean(dim=(1, 3))
    return out


def summarise(path):
    d = torch.load(path, weights_only=False)
    conds = [c for c in ("clean", "train", "heavy60") if c in d]
    res = {"meta": {k: v for k, v in d["meta"].items() if not k.startswith("verify")},
           "verify_max_rel_err_fp32": max(e[1] for c in conds for e in d["meta"][f"verify_rel_err_{c}"].values()),
           "n": len(d["clean"]["tasks"]), "pos_l2": {c: d[c]["pos_l2"].mean().item() for c in conds}, "layers": {}}
    assert all(d[c]["tasks"] == d["clean"]["tasks"] for c in conds)
    for layer in LAYERS:
        ps = {c: per_sample(d[c], layer) for c in conds}
        L = {}
        for c in conds:
            L[c] = {k: v.mean().item() for k, v in ps[c].items() if v.ndim == 1}
            L[c]["ext_enrich"] = L[c]["ext_frac_scene"] / L[c]["ext_token_frac"]
            L[c]["ext_frac_by_step"] = ps[c]["ext_frac_by_step"].mean(0).tolist()
        for c in conds:
            if c == "clean":
                continue
            delta = {}
            for k in ("ext_abs", "ext_frac_scene", "ctok_ext", "ctok_wrist", "registers", "camtoken", "traj",
                      "entropy", "entropy_norm"):
                x = ps[c][k] - ps["clean"][k]
                delta[k] = [x.mean().item(), (x.std() / len(x) ** 0.5).item()]
            hd = (ps[c]["ext_frac_by_head"] - ps["clean"]["ext_frac_by_head"]).mean(0)
            delta["ext_frac_scene_by_head"] = hd.tolist()
            sd = (ps[c]["ext_frac_by_step"] - ps["clean"]["ext_frac_by_step"]).mean(0)
            delta["ext_frac_scene_by_step"] = sd.tolist()
            x = (ps[c]["ext_frac_scene"] - ps["clean"]["ext_frac_scene"]).nan_to_num()
            delta["frac_samples_ext_down"] = (x < 0).double().mean().item()
            L[f"delta_{c}"] = delta
        res["layers"][layer] = L
    return res


def plot(summ, out):
    x = np.arange(len(LAYERS))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for (label, name), col in zip(MODELS, COLORS):
        if label not in summ:
            continue
        L = summ[label]["layers"]
        clean = [L[l]["clean"]["ext_frac_scene"] for l in LAYERS]
        heavy = [L[l]["heavy60"]["ext_frac_scene"] for l in LAYERS]
        axes[0].plot(x, clean, "-o", color=col, lw=2, ms=6, label=f"{name} clean")
        axes[0].plot(x, heavy, "--s", color=col, lw=2, ms=6, mfc="white", label=f"{name} heavy60")
        dm = [100 * L[l]["delta_heavy60"]["ext_frac_scene"][0] for l in LAYERS]
        se = [100 * L[l]["delta_heavy60"]["ext_frac_scene"][1] for l in LAYERS]
        axes[1].errorbar(x, dm, yerr=se, fmt="-o", color=col, lw=2, ms=6, capsize=3, label=name)
        de = [L[l]["delta_heavy60"]["entropy"][0] for l in LAYERS]
        dse = [L[l]["delta_heavy60"]["entropy"][1] for l in LAYERS]
        axes[2].errorbar(x, de, yerr=dse, fmt="-o", color=col, lw=2, ms=6, capsize=3, label=name)
        axes[0].plot(x, [L[l]["clean"]["ext_token_frac"] for l in LAYERS], ":", color=col, lw=1.2)
    axes[0].plot([], [], ":", color="#52514e", lw=1.2, label="dotted: share of keys from cams 0+1 (uniform attention)")
    axes[0].set_title("Attention on cams 0+1 / attention on all scene tokens")
    axes[1].set_title("Δ external share, heavy60 − clean (pp, ±SE)")
    axes[2].set_title("Δ attention entropy, heavy60 − clean (nats, ±SE)")
    for ax in axes:
        ax.set_xticks(x, LAYERS)
        ax.grid(axis="y", color="#e5e4e0", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.axvspan(1.5, 5.5, color="#f2f1ec", zorder=0)
    for ax in axes[1:]:
        ax.axhline(0, color="#52514e", lw=1)
    axes[0].legend(fontsize=6.5, ncol=1, frameon=False, loc="upper right")
    axes[1].legend(fontsize=8, frameon=False)
    n = next(iter(summ.values()))["n"]
    fig.text(0.5, -0.02, "ca = cross-attention to dense tokens; sa = shared self-attention (shaded) over FPS tokens; "
             f"pos/rot = output heads. Averaged over {n} val samples, 5 flow steps, 6 heads, 2 trajectory queries.",
             ha="center", fontsize=8, color="#52514e")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"cam_attn_external_share.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(MODELS), figsize=(16, 4), sharey=True)
    groups = [("ext_abs", "scene cams 0+1", "#2a78d6"), ("wrist_abs", "scene cams 2+3", "#eb6834"),
              ("ctok_ext", "C-tokens 0+1", "#1baf7a"), ("ctok_wrist", "C-tokens 2+3", "#eda100"),
              ("registers", "registers", "#e87ba4"), ("camtoken", "camera token", "#4a3aa7"),
              ("traj", "trajectory tokens", "#888780")]
    for ax, (label, name) in zip(axes, MODELS):
        if label not in summ:
            continue
        L = summ[label]["layers"]
        sl = LAYERS[2:]
        for j, cond in enumerate(("clean", "heavy60")):
            bottom = np.zeros(len(sl))
            xs = np.arange(len(sl)) + (j - 0.5) * 0.4
            for k, gname, col in groups:
                v = np.array([L[l][cond][k] for l in sl])
                ax.bar(xs, v, 0.38, bottom=bottom, color=col, edgecolor="white", lw=0.5,
                       hatch="//" if cond == "heavy60" else None, label=gname if j == 0 else None)
                bottom += v
        ax.set_xticks(np.arange(len(sl)), sl)
        ax.set_title(f"{name}\n(left clean, right hatched heavy60)", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("trajectory-query attention mass")
    axes[-1].legend(fontsize=7, frameon=False, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"cam_attn_sa_mass_breakdown.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="results/analysis/camera_attention")
    p.add_argument("--fig-dir", default="docs/research/experiments/2026-10-06/figures")
    a = p.parse_args()
    summ = {}
    for label, _ in MODELS:
        f = Path(a.in_dir) / f"{label}.pt"
        if f.exists():
            summ[label] = summarise(f)
    Path(a.in_dir, "summary.json").write_text(json.dumps(summ, indent=1))
    for label, s in summ.items():
        print(f"\n## {label}  n={s['n']}  fp32 verify err={s['verify_max_rel_err_fp32']:.1e}  pos_l2={s['pos_l2']}")
        print("layer | ext_frac clean | heavy | Δ±SE (pp) | train Δ (pp) | tokfrac | enrich c→h | ent c | Δent | "
              "ctokE c/Δ | reg c/Δ | traj c/Δ | %down")
        for l in LAYERS:
            L = s["layers"][l]
            c, h, dh = L["clean"], L["heavy60"], L["delta_heavy60"]
            dt = L.get("delta_train", {}).get("ext_frac_scene", [float("nan")])[0]
            print(f"{l} | {c['ext_frac_scene']:.3f} | {h['ext_frac_scene']:.3f} | "
                  f"{100*dh['ext_frac_scene'][0]:+.2f}±{100*dh['ext_frac_scene'][1]:.2f} | {100*dt:+.2f} | "
                  f"{c['ext_token_frac']:.3f}/{h['ext_token_frac']:.3f} | {c['ext_enrich']:.2f}→{h['ext_enrich']:.2f} | "
                  f"{c['entropy']:.2f} | {dh['entropy'][0]:+.3f}±{dh['entropy'][1]:.3f} | "
                  f"{c['ctok_ext']:.3f}/{dh['ctok_ext'][0]:+.3f} | {c['registers']:.3f}/{dh['registers'][0]:+.3f} | "
                  f"{c['traj']:.3f}/{dh['traj'][0]:+.3f} | {dh['frac_samples_ext_down']:.2f}")
    fig_dir = Path(a.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    if all("heavy60" in s["layers"]["ca0"] for s in summ.values()):
        plot(summ, fig_dir)


if __name__ == "__main__":
    main()
