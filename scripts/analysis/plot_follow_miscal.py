#!/usr/bin/env python3
"""Following coefficients, distances and figures from follow_miscal.py recordings (TODO 6)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from utils.plotting import configure_theme, save_figure

MODELS = {  # label -> (legend name, categorical slot colour), fixed order
    "exp3_200k": ("exp 3, 200k (EMA)", "#2a78d6"),
    "exp3_110k": ("exp 3, 110k (EMA)", "#eb6834"),
    "adaln1b_200k": ("1b, 200k (EMA)", "#1baf7a"),
    "controlA_200k": ("control A, 200k (raw)", "#eda100"),
    "fp": ("FP (full-patch, a6000_b64)", "#008300"),
    "fp_d8": ("FP-d8", "#4a3aa7"),
    "fp_r3d_refine": ("FP-R3D-refine", "#e34948"),
    "fp_r3d_direct": ("FP-R3D-direct", "#606B75"),
    "exp3_120000": ("exp 3, 120k (EMA)", "#2a78d6"),
    "exp4a_120000": ("exp 4a (pre-LN extractor), 120k (EMA)", "#eb6834"),
    "exp3_200000": ("exp 3, 200k (EMA)", "#2a78d6"),
    "exp4a_200000": ("exp 4a (pre-LN extractor), 200k (EMA)", "#eb6834"),
}
MAGS = (10, 20, 40, 60)
FIG = Path("docs/research/experiments/2026-10-06/figures")
RNG = np.random.default_rng(0)
PREFIX = ""


def unpack(r):
    """Positions (n, 2, 3), rotation matrices (n, 2, 3, 3) of pred and gt; N (n, 4, 4, 4)."""
    pred, gt = r["pred"][:, 0].double().numpy(), r["gt"][:, 0].double().numpy()
    rm = lambda q: Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(*q.shape[:-1], 3, 3)  # xyzw
    return pred[..., :3], gt[..., :3], rm(pred[..., 3:7]), rm(gt[..., 3:7]), r["N"].double().numpy()


def apply(N, x):
    """N (n, 4, 4) applied to points x (n, 2, 3)."""
    return np.einsum("nij,nhj->nhi", N[:, :3, :3], x) + N[:, None, :3, 3]


def lsq(d, us, idx=None):
    """Joint least-squares d ≈ sum_j a_j u_j pooled over samples/hands/xyz; d, u_j (n, 2, 3)."""
    idx = np.arange(len(d)) if idx is None else idx
    X = np.stack([u[idx].reshape(-1) for u in us], 1)
    return np.linalg.lstsq(X, d[idx].reshape(-1), rcond=None)[0]


def boot(d, us, n=500):
    """Point estimate and 95% bootstrap CI over samples for each coefficient."""
    est = lsq(d, us)
    bs = np.array([lsq(d, us, RNG.integers(0, len(d), len(d))) for _ in range(n)])
    lo, hi = np.percentile(bs, [2.5, 97.5], 0)
    return [(float(e), float(l), float(h)) for e, l, h in zip(est, lo, hi)]


def rotvec(R):
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(*R.shape[:-2], 3)


def analyse(rec):
    pc, gt, Rc, Rgt, _ = unpack(rec["clean"])
    tasks = np.array(rec["clean"]["tasks"])
    clean_rot = np.degrees(np.linalg.norm(rotvec(Rc @ np.swapaxes(Rgt, -1, -2)), axis=-1))
    out = {"clean": {"pos_l2_cm": float(100 * np.linalg.norm(pc - gt, axis=-1).mean()),
                     "rot_err_deg": float(clean_rot.mean()), "n": int(len(gt))}}
    for m in MAGS:
        for cond, cams in (("both", [0, 1]), ("cam0", [0]), ("cam1", [1]), ("rigid", [0])):
            if f"{cond}_{m}" not in rec:
                continue
            ph, g2, Rh, _, N = unpack(rec[f"{cond}_{m}"])
            assert np.allclose(g2, gt)
            us = [apply(N[:, j], gt) - gt for j in cams]
            d, ds = ph - gt, ph - pc
            # Rotation: does the predicted orientation rotate with N_j's rotation (about the gripper)?
            wN = [rotvec(np.broadcast_to(N[:, j, None, :3, :3], Rgt.shape)) for j in cams]
            wd = rotvec(Rh @ np.swapaxes(Rgt, -1, -2))
            wds = rotvec(Rh @ np.swapaxes(Rc, -1, -2))
            tgt = [gt] + [apply(N[:, j], gt) for j in cams]
            dist = np.stack([np.linalg.norm(ph - t, axis=-1) for t in tgt], -1)  # (n, 2, 1+len(cams))
            names = ["true"] + [f"N{j}p" for j in cams]
            r = {"alpha_vs_gt": dict(zip([f"a{j}" for j in cams], boot(d, us))),
                 "alpha_vs_clean_pred": dict(zip([f"a{j}" for j in cams], boot(ds, us))),
                 "beta_rot_vs_gt": dict(zip([f"b{j}" for j in cams], boot(wd, wN))),
                 "beta_rot_vs_clean_pred": dict(zip([f"b{j}" for j in cams], boot(wds, wN))),
                 "mean_dist_cm": {k: float(100 * dist[..., i].mean()) for i, k in enumerate(names)},
                 "closest_share": {k: float((dist.argmin(-1) == i).mean()) for i, k in enumerate(names)},
                 "mean_shift_from_clean_pred_cm": float(100 * np.linalg.norm(ds, axis=-1).mean()),
                 "mean_follow_target_offset_cm": {f"N{j}p": float(100 * np.linalg.norm(u, axis=-1).mean())
                                                  for j, u in zip(cams, us)},
                 "rot_err_deg": float(np.degrees(np.linalg.norm(wd, axis=-1)).mean())}
            if m == 60:
                r["per_task_alpha_vs_clean_pred"] = {
                    t: {"n": int((tasks == t).sum()),
                        **dict(zip([f"a{j}" for j in cams], lsq(ds, us, np.where(tasks == t)[0]).tolist()))}
                    for t in sorted(set(tasks))}
            out[f"{cond}_{m}"] = r
    return out


def fig_alpha(S, key, title, stem, letter):
    configure_theme()
    panels = [("both", 0, "both perturbed: α₀ (joint fit)"), ("both", 1, "both perturbed: α₁ (joint fit)"),
              ("cam0", 0, "only camera 0: α₀"), ("cam1", 1, "only camera 1: α₁"),
              ("rigid", 0, "all 4 cameras, same N: α")]
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6))
    off = np.linspace(-0.15, 0.15, len(S))
    for ax, (cond, j, t) in zip(axes, panels):
        for k, (label, s) in enumerate(S.items()):
            name, col = MODELS[label]
            v = np.array([s[f"{cond}_{m}"][key][f"{letter}{j}"] for m in MAGS])
            x = np.arange(len(MAGS)) + off[k]
            ax.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=col, lw=2, marker="o",
                        ms=5, capsize=2, label=name)
        ax.axhline(0, color="#4E5965", lw=0.8)
        ax.set_xticks(range(len(MAGS)), [f"{m}°\n{m} cm" for m in MAGS])
        ax.set_title(t, fontsize=10)
        ax.grid(axis="y", color="#D9DEE3", lw=0.8)
    lo = min(a.get_ylim()[0] for a in axes[:4])
    hi = max(a.get_ylim()[1] for a in axes[:4])
    for a in axes[:4]:
        a.set_ylim(lo, hi)
    axes[-1].set_ylim(min(0, axes[-1].get_ylim()[0]), max(1.0, axes[-1].get_ylim()[1]))
    axes[-1].axhline(1, color="#4E5965", lw=0.8, ls=":")
    axes[0].set_ylabel("following coefficient (0 = ignore, 1 = follow)")
    fig.text(0.01, -0.03, "Panels 1-4 share a zoomed y-axis; panel 5 is on a 0-1 scale.", fontsize=8, color="#4E5965")
    axes[-1].legend(frameon=False, fontsize=8.5)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, FIG / f"{PREFIX}{stem}")
    plt.close(fig)


def fig_dist(S):
    configure_theme()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, cond in zip(axes, ("both", "cam0", "cam1")):
        names = list(S[next(iter(S))][f"{cond}_60"]["closest_share"])
        w = 0.8 / len(S)
        for k, (label, s) in enumerate(S.items()):
            name, col = MODELS[label]
            v = [s[f"{cond}_60"]["closest_share"][n] for n in names]
            ax.bar(np.arange(len(names)) + (k - (len(S) - 1) / 2) * w, v, w * 0.92, color=col, label=name)
        ax.set_xticks(range(len(names)), [{"true": "true p", "N0p": "N₀p", "N1p": "N₁p"}[n] for n in names])
        ax.set_title({"both": "both cameras 60°/60cm", "cam0": "only camera 0", "cam1": "only camera 1"}[cond],
                     fontsize=10)
        ax.grid(axis="y", color="#D9DEE3", lw=0.8)
    axes[0].set_ylabel("share of (sample, hand) closest to")
    axes[-1].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Which target is the 60/60 prediction closest to?", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, FIG / f"{PREFIX}follow_closest_share")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="results/analysis/follow_miscal")
    p.add_argument("--fig-prefix", default="", help="prepended to figure file names")
    a = p.parse_args()
    global PREFIX
    PREFIX = a.fig_prefix
    S = {}
    for label in MODELS:
        f = Path(a.in_dir) / f"{label}.pt"
        if f.exists():
            rec = torch.load(f, weights_only=False)
            S[label] = analyse(rec)
            if "direction_check" in rec:
                S[label]["direction_check"] = rec["direction_check"]
    Path(a.in_dir, "summary.json").write_text(json.dumps(S, indent=1))
    fig_alpha(S, "alpha_vs_gt", "Position following α: (p̂ − p) ≈ α (N_j p − p), 95% bootstrap CI",
              "follow_alpha_pos", "a")
    fig_alpha(S, "beta_rot_vs_gt", "Orientation following β: log(R̂ Rᵀ) ≈ β log(R_Nj), 95% bootstrap CI",
              "follow_beta_rot", "b")
    fig_dist(S)
    for label, s in S.items():
        print(label, "clean", s["clean"])
        for m in MAGS:
            for c in ("both", "cam0", "cam1", "rigid"):
                if f"{c}_{m}" not in s:
                    continue
                r = s[f"{c}_{m}"]
                print(f"  {c}_{m}", {k: round(v[0], 4) for k, v in r["alpha_vs_gt"].items()},
                      {k: round(v[0], 4) for k, v in r["alpha_vs_clean_pred"].items()},
                      {k: round(v[0], 4) for k, v in r["beta_rot_vs_gt"].items()},
                      {k: round(v[0], 4) for k, v in r["beta_rot_vs_clean_pred"].items()},
                      {k: round(v, 2) for k, v in r["mean_dist_cm"].items()},
                      {k: round(v, 3) for k, v in r["closest_share"].items()},
                      round(r["mean_shift_from_clean_pred_cm"], 2), round(r["rot_err_deg"], 2))


if __name__ == "__main__":
    main()
