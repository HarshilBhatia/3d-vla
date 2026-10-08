#!/usr/bin/env python3
"""Per-layer, per-sample, per-step and per-camera delta_M statistics and figures from delta_m_per_layer.py recordings."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.plotting import configure_theme, save_figure

MODELS = {  # label -> (legend name, categorical slot colour), fixed order
    "exp3_200k": ("exp 3, 200k (EMA)", "#2a78d6"),
    "exp3_110k": ("exp 3, 110k (EMA)", "#eb6834"),
    "videopooled84k": ("VideoPooled 84k", "#1baf7a"),
    "randhist_warm50k": ("warm random history 50k", "#eda100"),
    "deltam140k": ("DeltaM 140k (no history)", "#e87ba4"),
    "fp": ("FP (full-patch)", "#2a78d6"),
    "fp_d8": ("FP-d8", "#eb6834"),
    "fp_r3d_refine": ("FP-R3D-refine", "#1baf7a"),
}
CALLS = ["init", "CA", "SA0", "SA1", "SA2", "SA3"]  # head call order per denoising step; "init" is predicted but unused in the layer-wise path
LAYERS = CALLS[1:]
CAMS = [0, 1]
EYE = torch.eye(6, dtype=torch.float64)


def fro(x):
    return x.flatten(-2).norm(dim=-1)


def log_angle(r):
    """||log R||_F and max principal angle (deg) of rotation matrices R (..., 6, 6)."""
    th = torch.linalg.eigvals(r).angle().abs()
    return th.pow(2).sum(-1).sqrt(), th.max(-1).values.rad2deg()


def task_share(v, tasks):
    """Fraction of across-sample variance of v (N, D) explained by task means (eta^2)."""
    v = v.reshape(len(v), -1)
    g = v.mean(0)
    total = (v - g).pow(2).sum(-1).mean()
    tasks = np.asarray(tasks)
    between = sum((tasks == t).sum() * (v[torch.from_numpy(tasks == t)].mean(0) - g).pow(2).sum()
                  for t in np.unique(tasks)) / len(v)
    if total.sqrt() < 1e-3:  # no across-sample variance: the share is undefined
        return float("nan"), float(total.sqrt())
    return float(between / total), float(total.sqrt())


def ms(x):
    return [float(x.mean()), float(x.std())]


def analyse(rec, n_first):
    out = {}
    clean, heavy = rec["clean"].double(), rec["heavy60"].double()  # (N, steps, calls, ncam, 6, 6)
    assert clean.shape[2] == len(CALLS), clean.shape
    tasks = rec["tasks"]
    out["n_samples"], out["n_tasks"] = clean.shape[0], len(set(tasks))
    out["init_eq_CA_max"] = float((clean[:, :, 0] - clean[:, :, 1]).abs().max())
    out["CA_eq_SA0_max"] = float((clean[:, :, 1] - clean[:, :, 2]).abs().max())
    out["wrist_cams_dist_from_I_max"] = float(fro(clean[:, :, :, 2:] - EYE).max())
    c, h = clean[:, :, 1:, :2], heavy[:, :, 1:, :2]  # (N, S, L, 2, 6, 6)
    dist = fro(c - EYE)  # (N, S, L, 2)
    out["dist_from_I"] = {f"cam{k}": {l: ms(dist[:, :, j, k].mean(1)) for j, l in enumerate(LAYERS)} for k in CAMS}
    out["dist_from_I_first320_allcalls"] = [float(fro(clean[:n_first, :, :, k] - EYE).mean()) for k in CAMS]
    out["dist_from_I_heavy"] = {f"cam{k}": {l: ms(fro(h - EYE)[:, :, j, k].mean(1)) for j, l in enumerate(LAYERS)} for k in CAMS}
    dd = fro(c[:, :, 1:] - c[:, :, :-1])
    ang, mx = log_angle(c[:, :, 1:] @ c[:, :, :-1].transpose(-1, -2))
    out["depth_change"] = {f"cam{k}": {f"{LAYERS[j]}->{LAYERS[j + 1]}": {
        "fro": ms(dd[:, :, j, k]), "log_fro": ms(ang[:, :, j, k]), "max_angle_deg": ms(mx[:, :, j, k])}
        for j in range(len(LAYERS) - 1)} for k in CAMS}
    ang0, mx0 = log_angle(c)
    out["log_fro_from_I"] = {f"cam{k}": {l: ms(ang0[:, :, j, k]) for j, l in enumerate(LAYERS)} for k in CAMS}
    vm = c.mean(1)  # step-mean per sample (N, L, 2, 6, 6)
    out["across_sample"] = {}
    for name, sl in [("full", slice(None)), ("first320", slice(0, n_first))]:
        out["across_sample"][name] = {f"cam{k}": {l: dict(zip(
            ["task_share", "rms_spread", "task_share_scalar"],
            [*task_share(vm[sl, j, k], tasks[sl]), task_share(fro(vm[sl, j, k] - EYE)[:, None], tasks[sl])[0]]))
            for j, l in enumerate(LAYERS)} for k in CAMS}
    ch = fro(c - h)
    out["clean_vs_heavy"] = {f"cam{k}": {l: ms(ch[:, :, j, k]) for j, l in enumerate(LAYERS)} for k in CAMS}
    dev = fro(c - c.mean(1, keepdim=True)).pow(2).mean(1).sqrt()  # (N, L, 2)
    pw = fro(c[:, :, None] - c[:, None]).amax((1, 2))
    out["across_steps"] = {f"cam{k}": {l: {"rms_dev": ms(dev[:, j, k]), "max_pairwise": ms(pw[:, j, k]),
                                          "dist_from_I_per_step": dist[:, :, j, k].mean(0).tolist()}
                                      for j, l in enumerate(LAYERS)} for k in CAMS}
    cc = fro(c[..., 0, :, :] - c[..., 1, :, :])
    a01, m01 = log_angle(c[..., 0, :, :] @ c[..., 1, :, :].transpose(-1, -2))
    out["cam0_vs_cam1"] = {l: {"fro": ms(cc[:, :, j]), "log_fro": ms(a01[:, :, j]), "max_angle_deg": ms(m01[:, :, j])}
                           for j, l in enumerate(LAYERS)}
    out["pos_l2"] = {cnd: float(rec[f"pos_l2_{cnd}"].mean()) for cnd in ("clean", "heavy60")}
    return out


def panel_fig(stats, getter, ylabel, title, stem, xlabels=LAYERS, per_cam=True, note=None):
    configure_theme()
    ncol = 2 if per_cam else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 3.6), sharey=True, squeeze=False)
    x = np.arange(len(xlabels))
    for ci, ax in enumerate(axes[0]):
        for label, s in stats.items():
            name, col = MODELS[label]
            m, sd = np.array(getter(s, ci)).T
            ax.plot(x, m, color=col, lw=2, marker="o", ms=5, label=name)
            ax.fill_between(x, np.clip(m - sd, 0, None), m + sd, color=col, alpha=0.12, lw=0)  # quantities are non-negative
        ax.set_xticks(x, xlabels)
        ax.set_xlabel("delta_M prediction (before block)")
        ax.grid(axis="y", color="#D9DEE3", lw=0.8)
        ax.set_title(f"camera {ci}" if per_cam else "")
    axes[0][0].set_ylabel(ylabel)
    axes[0][-1].legend(frameon=False, fontsize=8.5)
    fig.suptitle(title, fontweight="bold")
    if note:
        fig.text(0.01, -0.04, note, fontsize=8, color="#4E5965")
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="results/analysis/delta_m_per_layer")
    p.add_argument("--fig-dir", default="docs/research/experiments/2026-10-06/figures")
    p.add_argument("--first", type=int, default=320)
    p.add_argument("--fig-prefix", default="dm_layer")
    a = p.parse_args()
    stats = {}
    for label in MODELS:
        f = Path(a.in_dir) / f"{label}.pt"
        if f.exists():
            stats[label] = analyse(torch.load(f, weights_only=False), a.first)
    (Path(a.in_dir) / "summary.json").write_text(json.dumps(stats, indent=1))
    fd = Path(a.fig_dir)
    band = "line = mean over samples, band = +/- 1 std; clean cameras; CA and SA0 see the same C-tokens"
    panel_fig(stats, lambda s, k: list(s["dist_from_I"][f"cam{k}"].values()), "||ΔM − I||_F",
              "delta_M distance from identity per layer", fd / f"{a.fig_prefix}_dist_from_identity", note=band)
    tr = [f"{LAYERS[j]}→{LAYERS[j + 1]}" for j in range(len(LAYERS) - 1)]
    panel_fig(stats, lambda s, k: [v["log_fro"] for v in s["depth_change"][f"cam{k}"].values()],
              "||log(ΔM_l · ΔM_prevᵀ)||_F (rad)", "delta_M change between consecutive layers",
              fd / f"{a.fig_prefix}_depth_change", xlabels=tr, note="Frobenius differences are in summary.json; " + band)
    panel_fig(stats, lambda s, k: [[v["task_share"], 0] for v in s["across_sample"]["full"][f"cam{k}"].values()],
              "share of across-sample variance\nexplained by task (η²)", "delta_M: how much of the per-sample variation is task",
              fd / f"{a.fig_prefix}_task_share", note="all 853 val samples, 13 tasks, step-mean delta_M as a 36-vector; chance ≈ 0.014; no point = no spread (RMS < 1e-3), share undefined")
    panel_fig(stats, lambda s, k: list(s["clean_vs_heavy"][f"cam{k}"].values()), "||ΔM_clean − ΔM_60/60||_F",
              "delta_M change under heavy 60°/60 cm miscalibration (same sample, same noise)",
              fd / f"{a.fig_prefix}_clean_vs_heavy", note="cameras 0 and 1 miscalibrated; band = +/- 1 std over samples and steps")
    panel_fig(stats, lambda s, k: [v["rms_dev"] for v in s["across_steps"][f"cam{k}"].values()],
              "RMS over flow steps of\n||ΔM_step − ΔM_mean||_F", "delta_M variation across the 5 denoising steps",
              fd / f"{a.fig_prefix}_across_steps", note=band)
    panel_fig(stats, lambda s, k: [v["log_fro"] for v in s["cam0_vs_cam1"].values()],
              "||log(ΔM_cam0 ΔM_cam1ᵀ)||_F (rad)", "camera 0 vs camera 1 delta_M", fd / f"{a.fig_prefix}_cam0_vs_cam1",
              per_cam=False, note=band)
    print(json.dumps({k: {"dist": v["dist_from_I"], "cvh": v["clean_vs_heavy"]} for k, v in stats.items()}, indent=1))


if __name__ == "__main__":
    main()
