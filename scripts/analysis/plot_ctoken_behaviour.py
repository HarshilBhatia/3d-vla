#!/usr/bin/env python3
"""Summarise ctoken_behaviour.py outputs (TODO 10) into summary.json, a printed table, and the per-layer relative-change figure."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

MODELS = [("base160k", "Base 160k (no history)"), ("deltam140k", "DeltaM 140k (no history)"),
          ("adaln1b_200k", "1b 200k (collapsed C)"), ("exp3_200k", "exp 3 200k (collapsed C)"),
          ("fp", "FP (full-patch)"), ("fp_d8", "FP-d8"), ("fp_r3d_refine", "FP-R3D-refine"), ("fp_r3d_direct", "FP-R3D-direct"),
          ("exp4a_120000", "exp 4a 120k (pre-LN extractor)"), ("exp3_120000", "exp 3 120k"),
          ("exp4a_200000", "exp 4a 200k (pre-LN extractor)"), ("exp3_200000", "exp 3 200k")]
LAYERS = ["in", "ca0", "ca1", "sa0", "sa1", "sa2", "sa3", "pos0", "pos1", "rot0", "rot1"]
PLOT_LAYERS = ["ca0", "ca1", "in", "sa0", "sa1", "sa2", "sa3", "pos0", "pos1", "", "rot0", "rot1"]
SERIES = [(0, "C-token cam 0", "#2a78d6", "-"), (1, "C-token cam 1", "#2a78d6", "--"), (2, "C-token cam 2", "#eb6834", "-"),
          (3, "C-token cam 3", "#eb6834", "--"), (4, "FPS tokens, cams 0+1", "#1baf7a", "-"),
          (5, "FPS tokens, cams 2+3", "#eda100", "-"), (6, "trajectory tokens", "#4a3aa7", "-")]


def eta2(v, labels):
    """Share of across-sample variance of v (N, D) explained by the label means."""
    v = v.double()
    g = v.mean(0)
    total = (v - g).pow(2).sum(-1).mean()
    if total.sqrt() < 1e-4:
        return float("nan")
    labels = np.asarray(labels)
    between = sum((labels == t).sum() * (v[torch.from_numpy(labels == t)].mean(0) - g).pow(2).sum()
                  for t in np.unique(labels)) / len(v)
    return float(between / total)


def summarise(d):
    T, ncam = d["T"], d["ncam"]
    out = {"n": len(d["tasks"]), "pos_l2_cm": {k: 100 * v for k, v in d["pos_l2"].items()}, "meta": d["meta"]}
    c = d["ctok"]["clean"].double()
    out["content"] = {
        "spread_per_cam": [float(c[:, j].std(0).mean()) for j in range(ncam)],
        "norm_per_cam": [float(c[:, j].norm(dim=-1).mean()) for j in range(ncam)],
        "pair_dist": {f"{i}-{j}": float((c[:, i] - c[:, j]).norm(dim=-1).mean()) for i in range(ncam) for j in range(i + 1, ncam)},
        "eta2_task": [eta2(c[:, j], d["tasks"]) for j in range(ncam)],
        "eta2_group": [eta2(c[:, j], d["groups"]) for j in range(ncam)],
        "heavy_minus_clean_max": float((d["ctok"]["heavy60"].double() - c).abs().max()),
    }
    out["content_by_layer"] = {}
    for n, x in d["ctok_layers"]["clean"].items():
        x = x.double().mean(1)
        out["content_by_layer"][n] = {"spread_per_cam": [float(x[:, j].std(0).mean()) for j in range(ncam)],
                                      "eta2_task": [eta2(x[:, j], d["tasks"]) for j in range(ncam)],
                                      "eta2_group": [eta2(x[:, j], d["groups"]) for j in range(ncam)],
                                      "cam_pair_0_1": float((x[:, 0] - x[:, 1]).norm(dim=-1).mean()),
                                      "cam_pair_0_2": float((x[:, 0] - x[:, 2]).norm(dim=-1).mean())}
    out["cpos_shift_cm"] = {cond: [100 * float((d["cpos"][cond] - d["cpos"]["clean"])[:, j].norm(dim=-1).mean()) for j in range(ncam)]
                            for cond in d["cpos"] if cond != "clean"}
    out["rel_change"] = {cond: {n: np.nanmean(v.mean(1).numpy(), 0).tolist() for n, v in dd.items()} for cond, dd in d["rel"].items()}
    out["attn"] = {}
    for cond, dd in d["attn"].items():
        out["attn"][cond] = {}
        for n, m in dd.items():
            m = m.double().mean(1)
            traj = m[:, :T].mean(1)
            ctr = {"traj_to_ctok_per_cam": traj[:, 4:8].mean(0).tolist(), "traj_to_ctok_total": float(traj[:, 4:8].sum(-1).mean())}
            for j in range(ncam):
                r = m[:, T + j]
                other = r[:, 4:8].sum(-1) - r[:, 4 + j]
                ctr[f"C{j}"] = {"ext_fps": float(r[:, 0:2].sum(-1).mean()), "wrist_fps": float(r[:, 2:4].sum(-1).mean()),
                                "own_C": float(r[:, 4 + j].mean()), "other_C": float(other.mean()),
                                "registers": float(r[:, 8].mean()), "camtoken": float(r[:, 9].mean()), "traj": float(r[:, 10].mean())}
            out["attn"][cond][n] = ctr
    return out


def figure(summ, path):
    present = [(lb, nm) for lb, nm in MODELS if lb in summ]
    fig, axes = plt.subplots(1, len(present), figsize=(4.4 * len(present), 4.2), sharey=True, squeeze=False)
    x = np.arange(len(PLOT_LAYERS))
    ticks = ["SA in" if l == "in" else l for l in PLOT_LAYERS]
    for ax, (lb, nm) in zip(axes[0], present):
        rc = summ[lb]["rel_change"]["heavy60"]
        for k, name, col, ls in SERIES:
            y = np.array([rc[l][k] if l in rc else np.nan for l in PLOT_LAYERS], dtype=float)
            ax.plot(x, y, ls, color=col, lw=2, marker="o", ms=4, label=name)
        ax.set_xticks(x, ticks, rotation=45)
        ax.set_title(nm, fontsize=10)
        ax.grid(axis="y", color="#e5e4e0", lw=0.8)
        ax.axvspan(1.5, 6.5, color="#f2f1ec", zorder=0)
        ax.axvline(9, color="#c3c2b7", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0][0].set_ylabel("‖h_heavy60 − h_clean‖ / ‖h_clean‖")
    axes[0][-1].legend(fontsize=8, frameon=False, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    fig.suptitle("Relative change of each token type under heavy 60°/60 cm, per layer (same sample, same noise)", fontsize=11)
    fig.text(0.5, -0.06, "ca = cross-attention (trajectory only); SA in = input to shared self-attention (shaded: SA in to sa3); pos and rot are parallel output heads, both fed by sa3. "
             "Mean over 853 val samples and 5 flow steps.", ha="center", fontsize=8, color="#52514e")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{path}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="results/analysis/ctoken_behaviour")
    p.add_argument("--fig", default="docs/research/experiments/2026-10-06/figures/ctoken_rel_change")
    a = p.parse_args()
    summ = {}
    for lb, _ in MODELS:
        f = Path(a.in_dir) / f"{lb}.pt"
        if f.exists():
            summ[lb] = summarise(torch.load(f, weights_only=False))
    Path(a.in_dir, "summary.json").write_text(json.dumps(summ, indent=1))
    for lb, s in summ.items():
        c = s["content"]
        print(f"\n## {lb} n={s['n']}")
        print(" pos_l2 cm:", {k: round(v, 2) for k, v in s["pos_l2_cm"].items()})
        print(" content spread/cam", [round(v, 4) for v in c["spread_per_cam"]], "norm", [round(v, 2) for v in c["norm_per_cam"]],
              "heavy-clean max", f"{c['heavy_minus_clean_max']:.1e}")
        print(" pair dist", {k: round(v, 3) for k, v in c["pair_dist"].items()})
        print(" eta2 task", [round(v, 2) for v in c["eta2_task"]], "group", [round(v, 2) for v in c["eta2_group"]])
        for n, v in s["content_by_layer"].items():
            print(f"   layer {n}: spread", [round(x, 3) for x in v["spread_per_cam"]], "eta2 task", [round(x, 2) for x in v["eta2_task"]],
                  "group", [round(x, 2) for x in v["eta2_group"]], "d01", round(v["cam_pair_0_1"], 3), "d02", round(v["cam_pair_0_2"], 3))
        print(" cpos shift cm", {k: [round(x, 1) for x in v] for k, v in s["cpos_shift_cm"].items()})
        print(" rel change heavy60 [C0 C1 C2 C3 FPSext FPSwr traj]:")
        for n in LAYERS:
            if n in s["rel_change"]["heavy60"]:
                print(f"   {n}", [round(x, 3) for x in s["rel_change"]["heavy60"][n]], " train", [round(x, 3) for x in s["rel_change"]["train"][n]])
        for cond in ("clean", "heavy60"):
            for n, v in s["attn"][cond].items():
                cs = " ".join(f"C{j}:ext{v[f'C{j}']['ext_fps']:.2f}/wr{v[f'C{j}']['wrist_fps']:.2f}/own{v[f'C{j}']['own_C']:.2f}/oth{v[f'C{j}']['other_C']:.2f}/reg{v[f'C{j}']['registers']:.2f}" for j in range(4))
                print(f"   attn {cond} {n}: traj->C {v['traj_to_ctok_total']:.3f} {[round(x, 3) for x in v['traj_to_ctok_per_cam']]} | {cs}")
    figure(summ, a.fig)


if __name__ == "__main__":
    main()
