#!/usr/bin/env python3
"""Statistics and figures for the 1b camera-adaLN signal recorded by adaln_signal.py."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.plotting import configure_theme, save_figure

LAYERS = ["CA+SA0", "SA1", "SA2", "SA3", "heads"]  # signal index: computed before CA/SA0, then after each SA layer
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
COND_COL = {"train": "#2a78d6", "heavy60": "#eb6834"}


def eta2(v, tasks):
    """Share of across-sample variance of v (N, D) explained by task; nan when there is no spread."""
    g = v.mean(0)
    total = (v - g).pow(2).sum(-1).mean()
    if total.sqrt() < 1e-4 * max(g.norm(), 1e-8):
        return float("nan")
    between = sum(m.sum() * (v[m].mean(0) - g).pow(2).sum() for m in (tasks == t for t in np.unique(tasks.numpy()))) / len(v)
    return float(between / total)


def cosines(v, tasks):
    """Mean pairwise cosine within task and across tasks, raw and after removing the grand mean."""
    out = {}
    for name, x in (("raw", v), ("centred", v - v.mean(0))):
        u = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        c = u @ u.T
        same = tasks[:, None] == tasks[None]
        off = ~torch.eye(len(v), dtype=torch.bool)
        out[name] = {"within_task": float(c[same & off].mean()), "across_task": float(c[~same].mean())}
    return out


def layer_stats(c, tasks):
    """c: (N, steps, L, C) camera term."""
    vm = c.mean(1)
    res = {}
    for j, l in enumerate(LAYERS):
        v = vm[:, j]
        res[l] = {"mean_norm": float(v.norm(dim=-1).mean()),
                  "rms_spread": float((v - v.mean(0)).pow(2).sum(-1).mean().sqrt()),
                  "task_eta2": eta2(v, tasks), "cosine": cosines(v, tasks)}
        res[l]["rel_spread"] = res[l]["rms_spread"] / res[l]["mean_norm"]
    return res


def analyse(rec):
    clean = rec["clean"]
    tasks = torch.tensor([sorted(set(clean["tasks"])).index(t) for t in clean["tasks"]])
    c, t, a = clean["cam"].double(), clean["time"].double(), clean["ada"].double()
    names, out = rec["ada_names"], {"n_samples": len(tasks), "n_tasks": int(tasks.max()) + 1}
    # 1: strength per AdaLN module
    out["modulation"] = {n: {"signal": LAYERS[int(a[0, 0, i, 6])],
                             "d_scale_over_scale_time": float((a[..., i, 0] / a[..., i, 1]).mean()),
                             "d_shift_over_shift_time": float((a[..., i, 3] / a[..., i, 4]).mean()),
                             "d_scale_over_scale_full": float((a[..., i, 0] / a[..., i, 2]).mean()),
                             "scale_time": float(a[..., i, 1].mean()), "d_scale": float(a[..., i, 0].mean()),
                             "shift_time": float(a[..., i, 4].mean()), "d_shift": float(a[..., i, 3].mean()),
                             "d_scale_over_scale_time_per_step": (a[..., i, 0] / a[..., i, 1]).mean(0).tolist()}
                         for i, n in enumerate(names)}
    # 2: across-sample variation of the camera term, with time_embs and the mean C-token as references
    out["camera_term"] = layer_stats(c, tasks)
    out["time_embs"] = layer_stats(t, tasks)["CA+SA0"]
    out["mean_C_token"] = layer_stats(clean["meanC"].double(), tasks)
    # 3: miscalibration sensitivity
    out["miscal"] = {}
    for cond in ("train", "heavy60"):
        if cond in rec:
            d = (c - rec[cond]["cam"].double()).norm(dim=-1)  # (N, steps, L)
            out["miscal"][cond] = {l: {"abs": float(d[:, :, j].mean()),
                                       "rel_to_norm": float((d[:, :, j] / c[:, :, j].norm(dim=-1)).mean()),
                                       "rel_to_spread": float(d[:, :, j].mean()) / max(out["camera_term"][l]["rms_spread"], 1e-12)}
                                   for j, l in enumerate(LAYERS)}
    # 4: depth
    u = c / c.norm(dim=-1, keepdim=True)
    out["depth_cosine"] = {f"{LAYERS[j]}->{LAYERS[j + 1]}": float((u[:, :, j] * u[:, :, j + 1]).sum(-1).mean())
                           for j in range(len(LAYERS) - 1)}
    # 5: denoising steps
    dev = (c - c.mean(1, keepdim=True)).norm(dim=-1).pow(2).mean(1).sqrt()  # (N, L)
    out["steps"] = {l: {"rms_dev_over_norm": float((dev[:, j] / c[:, :, j].norm(dim=-1).mean(1)).mean()),
                        "camera_norm_per_step": c[:, :, j].norm(dim=-1).mean(0).tolist(),
                        "time_norm_per_step": t[:, :, j].norm(dim=-1).mean(0).tolist()}
                    for j, l in enumerate(LAYERS)}
    out["pos_l2"] = {k: float(rec[k]["pos_l2"].mean()) for k in ("clean", "train", "heavy60") if k in rec}
    return out


def fig_modulation(s, stem):
    configure_theme()
    names = list(s["modulation"])
    short = [n.replace("_self_attn", "").replace("attn_layers.", "a").replace("ffw_layers.", "f").replace("cross_attn", "CA")
             .replace("self_attn", "SA").replace("position", "pos").replace("rotation", "rot") for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for key, col, lab in (("d_scale_over_scale_time", "#2a78d6", "scale"), ("d_shift_over_shift_time", "#eb6834", "shift")):
        ax.plot(x, [s["modulation"][n][key] for n in names], color=col, lw=2, marker="o", ms=5, label=lab)
    ax.set_xticks(x, short, rotation=60, ha="right", fontsize=8.5)
    ax.set_ylabel("‖full − time-only‖ / ‖time-only‖")
    ax.grid(axis="y", color="#D9DEE3", lw=0.8)
    ax.legend(frameon=False)
    ax.set_title("1b: how much the camera term changes each AdaLN's modulation", fontweight="bold")
    fig.text(0.01, -0.12, "mean over 853 val samples x 5 flow steps, clean cameras; CA layers and SA0 share the first signal",
             fontsize=8, color="#4E5965")
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def fig_layers(s, stem):
    configure_theme()
    x = np.arange(len(LAYERS))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    ct = s["camera_term"]
    axes[0].plot(x, [ct[l]["rel_spread"] for l in LAYERS], color="#2a78d6", lw=2, marker="o", label="camera term")
    axes[0].axhline(s["time_embs"]["rel_spread"], color="#52514e", lw=1.2, ls="--", label="time_embs (reference)")
    axes[0].set_ylabel("across-sample RMS spread / mean norm")
    axes[0].set_title("variation across samples")
    axes[1].plot(x, [ct[l]["task_eta2"] for l in LAYERS], color="#2a78d6", lw=2, marker="o")
    axes[1].set_ylabel("share explained by task (η²)")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("task share of that variation")
    for cond, d in s["miscal"].items():
        axes[2].plot(x, [d[l]["rel_to_spread"] for l in LAYERS], color=COND_COL[cond], lw=2, marker="o", label=cond)
    axes[2].set_ylabel("‖clean − miscal‖ / across-sample spread")
    axes[2].set_title("sensitivity to miscalibration")
    for ax in axes:
        ax.set_xticks(x, LAYERS)
        ax.grid(axis="y", color="#D9DEE3", lw=0.8)
        ax.set_xlabel("signal (layer it conditions)")
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[2].legend(frameon=False, fontsize=8.5)
    fig.suptitle("1b camera term MLP_c(mean C) per layer", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def fig_pca(rec, stem):
    configure_theme()
    clean = rec["clean"]
    names = sorted(set(clean["tasks"]))
    tasks = np.array([names.index(t) for t in clean["tasks"]])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, j in zip(axes, (1, len(LAYERS) - 1)):  # CA+SA0 term is constant (spread ~1e-6), so SA1 is the first informative one
        v = clean["cam"][:, :, j].double().mean(1)
        v = v - v.mean(0)
        _, sv, vh = torch.linalg.svd(v, full_matrices=False)
        z = (v @ vh[:2].T).numpy()
        ev = (sv[:2] ** 2 / (sv ** 2).sum()).tolist()
        for k, n in enumerate(names):
            m = tasks == k
            ax.scatter(z[m, 0], z[m, 1], s=14, color=PALETTE[k % 8], marker="o" if k < 8 else "^",
                       edgecolor="white", linewidth=0.4, label=n.replace("bimanual_", ""))
        ax.set_xlabel(f"PC1 ({ev[0]:.0%})")
        ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
        ax.set_title(f"{LAYERS[j]} signal (RMS spread {float(v.norm(dim=-1).pow(2).mean().sqrt()):.3g})")
        ax.grid(color="#D9DEE3", lw=0.6)
    axes[1].legend(frameon=False, fontsize=7.5, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("1b camera term, PCA of per-sample (step-mean) vectors, coloured by task (the CA+SA0 term is constant and not shown)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rec", default="results/analysis/adaln_signal/exp1b_200k.pt")
    p.add_argument("--fig-dir", default="docs/research/experiments/2026-10-06/figures")
    a = p.parse_args()
    rec = torch.load(a.rec, weights_only=False)
    s = analyse(rec)
    Path(a.rec).with_suffix(".json").write_text(json.dumps(s, indent=1))
    fd = Path(a.fig_dir)
    fig_modulation(s, fd / "adaln_1b_modulation_strength")
    fig_layers(s, fd / "adaln_1b_camera_term_layers")
    fig_pca(rec, fd / "adaln_1b_camera_term_pca")


if __name__ == "__main__":
    main()
