#!/usr/bin/env python3
"""What the learned delta_M does to the 3D RoPE kernel: displacement kernels, 6x6 structure and miscalibration sensitivity."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.plotting import configure_theme, save_figure

DIM = 120  # embedding_dim of every model here; 40 dims (20 bands) per axis
OMEGA = torch.exp(torch.arange(0, DIM // 3, 2, dtype=torch.float64) * (-math.log(10000.0) / (DIM // 3)))  # rad/m
WS = torch.tensor([[-0.2323, -0.7035, 0.7092], [0.8962, 0.4811, 1.5785]], dtype=torch.float64)  # workspace_normalizer
COMP = ["cos x", "cos y", "cos z", "sin x", "sin y", "sin z"]
CALL = {"CA": 1, "SA1": 3, "SA2": 4, "SA3": 5}  # index into the 6 recorded calls (init, CA, SA0, SA1, SA2, SA3)
COLS = {"identity": "#52514e", "CA": "#2a78d6", "SA1": "#eb6834", "SA2": "#1baf7a", "SA3": "#eda100"}


def basis(p):
    """(..., 3) metres -> (..., bands, 6) [cos x, cos y, cos z, sin x, sin y, sin z], as _compute_sincos_base."""
    a = p[..., None, :] * OMEGA[:, None]
    return torch.cat([a.cos(), a.sin()], -1)


def mix(f, dm):
    """Key-side mixing exactly as _apply_delta_M: out = delta_M @ feat per band."""
    return torch.einsum("...ci,ji->...cj", f, dm)


def pair_op(q, p, dm):
    """Per band and axis block, (alpha, beta) of the 2x2 relative operator M(cq,sq)^T M(ck,sk); query plain, key mixed."""
    fq, fk = basis(q), mix(basis(p), dm)
    cq, sq, ck, sk = fq[..., :3], fq[..., 3:], fk[..., :3], fk[..., 3:]
    return cq * ck + sq * sk, cq * sk - sq * ck  # (..., bands, 3)


def kernel(q, p, dm):
    """Even RoPE logit factor: mean over bands of alpha, per axis block (..., 3)."""
    return pair_op(q, p, dm)[0].mean(-2)


def project_so(m):
    u, _, vh = torch.linalg.svd(m)
    r = u @ vh
    if torch.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vh
    return r


def planes(dm):
    """Rotation angles (deg, one per conjugate pair or real eigenvalue) and the 6-component loading of each invariant plane."""
    w, v = torch.linalg.eig(dm)
    ang = w.angle()
    out = []
    for i in torch.argsort(-ang).tolist():
        if ang[i] < -1e-6:
            continue
        load = (v[:, i].abs() ** 2) * (2 if ang[i] > 1e-6 else 1)
        out.append({"angle_deg": float(ang[i].rad2deg()), "loading": {c: round(float(x), 3) for c, x in zip(COMP, load)}})
    return out


def blocks(dm):
    """Frobenius energy of delta_M - I: cos->cos, sin->sin, cos<->sin; and the 3x3 out-axis x in-axis 2x2-block norms."""
    e = dm - torch.eye(6, dtype=dm.dtype)
    idx = lambda a: [a, a + 3]
    return {"cc": float(e[:3, :3].norm()), "ss": float(e[3:, 3:].norm()), "cs_sc": float(torch.cat([e[:3, 3:], e[3:, :3]]).norm()),
            "axis_block_norm": [[float(dm[np.ix_(idx(a), idx(b))].norm()) for b in range(3)] for a in range(3)]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="results/analysis/delta_m_per_layer")
    p.add_argument("--out-dir", default="results/analysis/rope_kernel")
    p.add_argument("--fig-dir", default="docs/research/experiments/2026-10-06/figures")
    a = p.parse_args()
    g = torch.Generator().manual_seed(0)
    dms, samples = {}, {}
    for model in ("exp3_200k", "videopooled84k"):
        rec = torch.load(Path(a.in_dir) / f"{model}.pt", weights_only=False)
        c = rec["clean"].double()  # (N, steps, calls, ncam, 6, 6)
        for l, k in CALL.items():
            dms[f"{model}/{l}"] = project_so(c[:, :, k, 0].mean((0, 1)))
        tasks = rec["tasks"]
        pick = [tasks.index(t) for t in sorted(set(tasks))[:3]]
        samples[model] = {f"{tasks[i]}": c[i, :, CALL["SA1"], 0].mean(0) for i in pick}
    out = {"omega_rad_per_m": OMEGA.tolist(), "wavelength_m": (2 * math.pi / OMEGA).tolist()}
    # 2: structure
    out["structure"] = {k: {"dist_from_I": float((v - torch.eye(6, dtype=v.dtype)).norm()), "planes": planes(v), **blocks(v)}
                        for k, v in dms.items()}
    # 1: displacement kernels, query at the origin and averaged over workspace queries
    dist = torch.tensor([0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60], dtype=torch.float64)
    grid = torch.linspace(-0.6, 0.6, 241, dtype=torch.float64)
    qs = WS[0] + (WS[1] - WS[0]) * torch.rand(128, 3, generator=g, dtype=torch.float64)
    dirs = torch.randn(32, 3, generator=g, dtype=torch.float64)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    eye = torch.eye(6, dtype=torch.float64)
    names = ["identity"] + [f"exp3_200k/{l}" for l in CALL] + [f"videopooled84k/{l}" for l in ("CA", "SA1")]
    get = lambda n: eye if n == "identity" else dms[n]
    kern = {}
    for n in names:
        dm = get(n)
        r = {}
        for ai, ax in enumerate("xyz"):
            d = torch.zeros(len(grid), 3, dtype=torch.float64)
            d[:, ai] = grid
            k0 = kernel(torch.zeros(1, 3, dtype=torch.float64), d, dm)  # (G, 3)
            kw = kernel(qs[None], qs[None] + d[:, None], dm).mean(1)
            r[ax] = {"origin_total": k0.mean(-1).tolist(), "origin_block": k0[:, ai].tolist(),
                     "ws_total": kw.mean(-1).tolist(), "ws_block": kw[:, ai].tolist()}
            tot = kw.mean(-1)
            r[ax]["ws_peak_at_m"] = float(grid[tot.argmax()])
            r[ax]["ws_value_at_0"] = float(tot[len(grid) // 2])
            r[ax]["ws_drop_0_to_60cm"] = float(tot[len(grid) // 2] - 0.5 * (tot[0] + tot[-1]))
            o = k0.mean(-1)
            r[ax]["origin_peak_at_m"] = float(grid[o.argmax()])
            r[ax]["origin_drop_0_to_60cm"] = float(o[len(grid) // 2] - 0.5 * (o[0] + o[-1]))
        rnd = kernel(torch.zeros(1, 1, 3, dtype=torch.float64), dist[:, None, None] * dirs[None], dm).mean(-1)  # (D, dirs)
        r["random_dirs_origin"] = {f"{float(x) * 100:.0f}cm": [float(rnd[i].mean()), float(rnd[i].std())] for i, x in enumerate(dist)}
        # top band phase and gain along x from the origin
        al, be = pair_op(torch.zeros(1, 3, dtype=torch.float64), torch.stack([dist, 0 * dist, 0 * dist], -1), dm)
        r["top_band_x"] = {"phase_rad": torch.atan2(be[:, 0, 0], al[:, 0, 0]).tolist(),
                           "gain": torch.sqrt(al[:, 0, 0] ** 2 + be[:, 0, 0] ** 2).tolist(), "plain_phase": dist.tolist()}
        kern[n] = r
    out["kernel"] = {n: {ax: {k: v for k, v in r[ax].items() if not isinstance(v, list)} for ax in "xyz"} | {
        "random_dirs_origin": r["random_dirs_origin"], "top_band_x": r["top_band_x"]} for n, r in kern.items()}
    out["kernel_at_dist"] = {n: {ax: {f"{float(x) * 100:.0f}cm": kern[n][ax]["origin_total"][int(torch.argmin((grid - x).abs()))]
                                      for x in dist} for ax in "xyz"} for n in names}
    # sample-to-sample: SA1 kernels for individual samples vs the mean delta_M
    out["sample_kernel_maxdiff_SA1"] = {}
    for model, ss in samples.items():
        ref = dms[f"{model}/SA1"]
        d = torch.zeros(len(grid), 3, dtype=torch.float64)
        d[:, 0] = grid
        km = kernel(torch.zeros(1, 3, dtype=torch.float64), d, ref).mean(-1)
        out["sample_kernel_maxdiff_SA1"][model] = {t: float((kernel(torch.zeros(1, 3, dtype=torch.float64), d, project_so(m)).mean(-1) - km).abs().max())
                                                   for t, m in ss.items()}
    # 3: miscalibration, P' = R P + t (left-multiplied extrinsics), 60 deg about a random axis, 60 cm in a random direction
    n_p, n_t = 128, 128
    P = WS[0] + (WS[1] - WS[0]) * torch.rand(n_p, 3, generator=g, dtype=torch.float64)
    Q = WS[0] + (WS[1] - WS[0]) * torch.rand(n_p, 3, generator=g, dtype=torch.float64)
    ax_ = torch.randn(n_t, 3, generator=g, dtype=torch.float64)
    ax_ = ax_ / ax_.norm(dim=-1, keepdim=True)
    K = torch.zeros(n_t, 3, 3, dtype=torch.float64)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -ax_[:, 2], ax_[:, 1], -ax_[:, 0]
    K = K - K.transpose(1, 2)
    th = math.radians(60)
    R = torch.eye(3, dtype=torch.float64) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
    tdir = torch.randn(n_t, 3, generator=g, dtype=torch.float64)
    t = 0.6 * tdir / tdir.norm(dim=-1, keepdim=True)
    P2 = torch.einsum("tij,pj->tpi", R, P) + t[:, None]  # (T, P, 3)
    out["miscal"] = {"mean_point_displacement_m": float((P2 - P).norm(dim=-1).mean())}
    for n in names:
        dm = get(n)
        a0, b0 = pair_op(Q[None], P[None], dm)
        a1, b1 = pair_op(Q[None].expand(n_t, -1, -1), P2, dm)
        op = ((a1 - a0) ** 2 + (b1 - b0) ** 2).mean(-2)  # (T, P, 3): per-block operator change, mean over bands
        kch = (a1.mean(-2) - a0.mean(-2)).mean(-1).abs()  # even-kernel change, all blocks
        hi = ((a1[..., :3, :] - a0[..., :3, :]) ** 2 + (b1[..., :3, :] - b0[..., :3, :]) ** 2).sum(-1).mean(-1)  # top 3 bands
        out["miscal"][n] = {"op_change_rms_per_block": op.mean((0, 1)).sqrt().tolist(),
                            "op_change_rms_total": float(op.sum(-1).mean().sqrt()),
                            "op_change_rms_top3_bands": float(hi.mean().sqrt()),
                            "even_kernel_abs_change": float(kch.mean())}
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.out_dir) / "summary.json").write_text(json.dumps(out, indent=1))
    torch.save({"dms": dms, "grid": grid, "kern": kern}, Path(a.out_dir) / "kernels.pt")
    figures(kern, grid, dms, Path(a.fig_dir))
    print(json.dumps({k: out[k] for k in ("structure", "miscal", "kernel_at_dist", "sample_kernel_maxdiff_SA1")}, indent=1))


def figures(kern, grid, dms, fd):
    configure_theme()
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), sharey="row")
    x = grid.numpy() * 100
    for row, key in enumerate(("origin_total", "ws_total")):
        for ai, ax_name in enumerate("xyz"):
            ax = axes[row, ai]
            for n in ["identity"] + [f"exp3_200k/{l}" for l in CALL]:
                lab = n.split("/")[-1]
                ax.plot(x, kern[n][ax_name][key], color=COLS[lab], lw=2, ls="--" if lab == "identity" else "-",
                        label="identity (plain RoPE)" if lab == "identity" else f"exp 3 delta_M before {lab}")
            ax.axvline(0, color="#D9DEE3", lw=0.8)
            ax.grid(axis="y", color="#D9DEE3", lw=0.8)
            ax.set_title(f"key displaced along {ax_name}" + (", query at origin" if row == 0 else ", query in workspace (mean)"), fontsize=10)
            ax.set_xlabel("displacement (cm)")
        axes[row, 0].set_ylabel("RoPE logit factor\n(mean over bands and axis blocks)")
    axes[0, 2].legend(frameon=False, fontsize=8.5)
    fig.suptitle("exp 3 (200k, EMA): RoPE kernel vs key displacement, plain basis vs learned delta_M (camera 0)", fontweight="bold")
    fig.text(0.01, -0.02, "query uses the plain basis, scene key the delta_M-mixed one; SA0 uses the CA delta_M; even part (alpha) only",
             fontsize=8, color="#4E5965")
    fig.tight_layout()
    save_figure(fig, fd / "rope_kernel_displacement")
    plt.close(fig)
    keys = [f"exp3_200k/{l}" for l in CALL] + ["videopooled84k/CA", "videopooled84k/SA1"]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.9 * len(keys), 3.6), gridspec_kw={"wspace": 0.55})
    for ax, k in zip(axes, keys):
        im = ax.imshow(dms[k].numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(6), COMP, rotation=90, fontsize=8)
        ax.set_yticks(range(6), COMP, fontsize=8)
        ax.set_title(k.replace("exp3_200k", "exp 3").replace("videopooled84k", "VideoPooled"), fontsize=10)
        for i in range(6):
            for j in range(6):
                v = float(dms[k][i, j])
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(v) > 0.6 else "#0b0b0b")
    fig.colorbar(im, ax=axes, fraction=0.015)
    fig.suptitle("delta_M (camera 0, sample mean): row = output component, column = input component", fontweight="bold")
    save_figure(fig, fd / "rope_delta_m_heatmaps")
    plt.close(fig)


if __name__ == "__main__":
    main()
