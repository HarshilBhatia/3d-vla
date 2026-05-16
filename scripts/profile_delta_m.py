"""
Profile deltaM (6x6) vs deltaM_full (DxD) computation time and memory.

Two cost centers per mode:
  1. Prediction: camera_proj + camera_predictor (MLP) + antisymmetrize + matrix_exp
  2. RoPE mixing: per-token einsum applying delta_M to sin/cos features

Usage:
  python scripts/profile_delta_m.py [--device cuda] [--batch 32] [--ncam 3]
                                     [--n_tokens 2000] [--fps_tokens 400]
                                     [--emb_dim 120] [--n_warmup 20] [--n_runs 100]
"""
import argparse
import time
import torch
import torch.nn as nn


def make_timer(device):
    if device == "cuda":
        def timer():
            torch.cuda.synchronize()
            return time.perf_counter()
    else:
        timer = time.perf_counter
    return timer


def time_fn(fn, n_warmup, n_runs, timer, max_seconds=5.0):
    """Run fn n_warmup times, then time n_runs iterations (or fewer if time budget exceeded)."""
    with torch.no_grad():
        for _ in range(n_warmup):
            fn()
        t0 = timer()
        actual = 0
        deadline = t0 + max_seconds
        for _ in range(n_runs):
            fn()
            actual += 1
            if timer() > deadline:
                break
    t1 = timer()
    return (t1 - t0) / actual * 1e3  # ms per call


def mem_mb(device, fn):
    """Peak GPU memory delta (MB) from fn(); returns 0 on CPU."""
    if device != "cuda":
        return 0.0
    with torch.no_grad():
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        fn()
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
    return (after - before) / 1024**2


def time_and_mem(fn, n_warmup, n_runs, timer, device, max_seconds=5.0):
    ms = time_fn(fn, n_warmup, n_runs, timer, max_seconds)
    mb = mem_mb(device, fn)
    return ms, mb


# ---- deltaM 6x6 prediction ----

def predict_delta_m_6x6(cam_proj, cam_pred, per_img_feats):
    """Mirrors _predict_from_cam_feat (delta_m mode)."""
    h = cam_proj(per_img_feats)
    A_skew = cam_pred(h).reshape(*per_img_feats.shape[:-1], 6, 6)
    A = A_skew - A_skew.transpose(-1, -2)
    norm = torch.linalg.norm(A, ord='fro', dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    A = A * (norm.clamp(max=3.0) / norm)
    return torch.linalg.matrix_exp(A)  # (B, ncam, 6, 6)


# ---- deltaM_full DxD prediction ----

def predict_delta_m_full(cam_proj, cam_pred, per_img_feats, D):
    """Mirrors _predict_from_cam_feat (delta_m_full mode)."""
    h = cam_proj(per_img_feats)
    A_skew = cam_pred(h).reshape(*per_img_feats.shape[:-1], D, D)
    A = A_skew - A_skew.transpose(-1, -2)
    norm = torch.linalg.norm(A, ord='fro', dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    A = A * (norm.clamp(max=3.0) / norm)
    return torch.linalg.matrix_exp(A)  # (B, ncam, D, D)


# ---- RoPE einsum: per-token 6x6 (delta_m) ----

def apply_rope_6x6(base_feat, delta_M_tokens):
    """(B, N, d//6, 6) x (B, N, 6, 6) -> (B, N, d//6, 6)."""
    return torch.einsum('bnci,bnji->bncj', base_feat, delta_M_tokens)


# ---- RoPE einsum: per-token DxD (delta_m_full, current code) ----

def apply_rope_DxD(base_feat_flat, delta_M_tokens):
    """(B, N, D) x (B, N, D, D) -> (B, N, D). Requires pre-expanded per-token delta_M."""
    return torch.einsum('bni,bnji->bnj', base_feat_flat, delta_M_tokens)


# ---- RoPE: grouped matmul (delta_m_full, no expand) ----

def apply_rope_DxD_grouped(base_feat_flat, delta_M_per_cam, ncam):
    """(B, N, D) x (B, ncam, D, D) -> (B, N, D). No per-token expansion, direct batched GEMM."""
    B, N, D = base_feat_flat.shape
    P = N // ncam
    feat = base_feat_flat[:, :ncam * P].reshape(B, ncam, P, D)
    out = feat @ delta_M_per_cam.transpose(-1, -2)  # (B, ncam, P, D)
    return out.reshape(B, ncam * P, D)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--ncam', type=int, default=3)
    p.add_argument('--n_tokens', type=int, default=2000, help='dense rgb3d tokens')
    p.add_argument('--fps_tokens', type=int, default=400, help='fps scene tokens (M + ncam)')
    p.add_argument('--emb_dim', type=int, default=120)
    p.add_argument('--n_warmup', type=int, default=20)
    p.add_argument('--n_runs', type=int, default=100)
    args = p.parse_args()

    device = args.device
    B, ncam, C = args.batch, args.ncam, args.emb_dim
    N_dense = args.n_tokens
    N_fps = args.fps_tokens

    # D for delta_m_full: same formula as TransformerHead.__init__
    D = (C // 6) * 6  # e.g. 120 for emb_dim=120
    d_bin = C // 3 // 2  # d//6 bins (= dx//2 = C//3//2)
    # base_feat shape: (B, N, d_bin, 6) -- output of _compute_sincos_base

    print(f"\n=== deltaM profiling  ({device}, B={B}, ncam={ncam}, emb_dim={C}, D={D}) ===")
    print(f"  Dense rgb3d tokens : {N_dense}  ({N_dense//ncam} per cam)")
    print(f"  FPS scene tokens   : {N_fps}")
    print(f"  Runs: {args.n_runs} (warmup={args.n_warmup})")

    # ---- Build modules ----
    cam_proj = nn.Linear(C, C).to(device)
    cam_pred_6x6 = nn.Sequential(nn.Linear(C, C), nn.ReLU(), nn.Linear(C, 36)).to(device)
    cam_pred_full = nn.Sequential(nn.Linear(C, C), nn.ReLU(), nn.Linear(C, D * D)).to(device)

    # ---- Inputs ----
    per_img_feats = torch.randn(B, ncam, C, device=device)
    base_dense = torch.randn(B, N_dense, d_bin, 6, device=device)
    base_fps = torch.randn(B, N_fps, d_bin, 6, device=device)
    base_dense_flat = base_dense.reshape(B, N_dense, -1)  # (B, N_dense, D)
    base_fps_flat = base_fps.reshape(B, N_fps, -1)

    # cam_ids for expanding (B, ncam, ...) -> (B, N, ...)
    cam_ids_dense = torch.arange(ncam, device=device).repeat_interleave(
        (N_dense + ncam - 1) // ncam)[:N_dense]
    cam_ids_fps = torch.zeros(N_fps - ncam, dtype=torch.long, device=device)

    timer = make_timer(device)

    show_mem = (device == "cuda")
    hdr = f"{'Operation':<50} {'6x6 ms':>8} {'DxD ms':>8}"
    if show_mem:
        hdr += f"  {'6x6 MB':>8} {'DxD MB':>8}"
    print()
    print(hdr)
    print("-" * (len(hdr) + 2))

    def row(label, fn6, fnD):
        t6, m6 = time_and_mem(fn6, args.n_warmup, args.n_runs, timer, device)
        tD, mD = time_and_mem(fnD, args.n_warmup, args.n_runs, timer, device)
        line = f"  {label:<48} {t6:>8.3f} {tD:>8.3f}"
        if show_mem:
            line += f"  {m6:>8.1f} {mD:>8.1f}"
        print(line)
        return t6, tD

    # ---- matrix_exp only ----
    A_6x6 = torch.randn(B, ncam, 6, 6, device=device)
    A_6x6 = A_6x6 - A_6x6.transpose(-1, -2)
    A_DxD = torch.randn(B, ncam, D, D, device=device)
    A_DxD = A_DxD - A_DxD.transpose(-1, -2)

    row(f"matrix_exp (B,{ncam},6,6) vs (B,{ncam},{D},{D})",
        lambda: torch.linalg.matrix_exp(A_6x6),
        lambda: torch.linalg.matrix_exp(A_DxD))

    # ---- Full prediction (proj + MLP + exp) ----
    row("Prediction (proj + MLP + exp)",
        lambda: predict_delta_m_6x6(cam_proj, cam_pred_6x6, per_img_feats),
        lambda: predict_delta_m_full(cam_proj, cam_pred_full, per_img_feats, D))

    # ---- Expand (B,ncam,_,_) -> (B,N_dense,_,_) ----
    dM_6x6 = torch.randn(B, ncam, 6, 6, device=device)
    dM_DxD = torch.randn(B, ncam, D, D, device=device)

    row(f"Expand to dense N={N_dense} (index gather)",
        lambda: dM_6x6[:, cam_ids_dense, :, :],
        lambda: dM_DxD[:, cam_ids_dense, :, :])

    # ---- RoPE einsum — dense tokens ----
    dM_dense_6x6 = dM_6x6[:, cam_ids_dense, :, :]
    dM_dense_DxD = dM_DxD[:, cam_ids_dense, :, :]

    row(f"RoPE einsum dense N={N_dense}  (expand+einsum)",
        lambda: apply_rope_6x6(base_dense, dM_dense_6x6),
        lambda: apply_rope_DxD(base_dense_flat, dM_dense_DxD))

    row(f"RoPE einsum dense N={N_dense}  (grouped, no expand)",
        lambda: apply_rope_6x6(base_dense, dM_dense_6x6),           # 6x6 unchanged
        lambda: apply_rope_DxD_grouped(base_dense_flat, dM_DxD, ncam))

    # ---- RoPE einsum — fps tokens ----
    dM_fps_6x6 = torch.cat([dM_6x6[:, cam_ids_fps, :, :], dM_6x6], dim=1)
    dM_fps_DxD = torch.cat([dM_DxD[:, cam_ids_fps, :, :], dM_DxD], dim=1)

    row(f"RoPE einsum fps N={N_fps}",
        lambda: apply_rope_6x6(base_fps, dM_fps_6x6),
        lambda: apply_rope_DxD(base_fps_flat, dM_fps_DxD))

    # ---- Full pipeline per call ----
    def full_6x6():
        dM = predict_delta_m_6x6(cam_proj, cam_pred_6x6, per_img_feats)
        apply_rope_6x6(base_dense, dM[:, cam_ids_dense, :, :])
        dM_f = torch.cat([dM[:, cam_ids_fps, :, :], dM], dim=1)
        apply_rope_6x6(base_fps, dM_f)

    def full_DxD():
        dM = predict_delta_m_full(cam_proj, cam_pred_full, per_img_feats, D)
        apply_rope_DxD(base_dense_flat, dM[:, cam_ids_dense, :, :])
        dM_f = torch.cat([dM[:, cam_ids_fps, :, :], dM], dim=1)
        apply_rope_DxD(base_fps_flat, dM_f)

    def full_DxD_grouped():
        dM = predict_delta_m_full(cam_proj, cam_pred_full, per_img_feats, D)
        apply_rope_DxD_grouped(base_dense_flat, dM, ncam)
        # fps sparse part still needs per-token expand (mixed cam ids); dense is the expensive one
        dM_f = torch.cat([dM[:, cam_ids_fps, :, :], dM], dim=1)
        apply_rope_DxD(base_fps_flat, dM_f)

    print("-" * (len(hdr) + 2))
    t6, tD = row("Full pipeline  (current)", full_6x6, full_DxD)
    _,  tDg = row("Full pipeline  (grouped dense, no expand)", full_6x6, full_DxD_grouped)

    n_layers = 6   # 2 CA + 4 SA (default dynamic_rope_from_camtoken config)
    n_steps = 5    # denoise_timesteps=5
    print(f"\n  Estimated inference ({n_layers} layers × {n_steps} steps = {n_layers*n_steps} calls):")
    print(f"    6x6             : {t6  * n_layers * n_steps:7.1f} ms  ({t6  * n_layers * n_steps / 1000:.3f} s)")
    print(f"    DxD (current)   : {tD  * n_layers * n_steps:7.1f} ms  ({tD  * n_layers * n_steps / 1000:.3f} s)")
    print(f"    DxD (grouped)   : {tDg * n_layers * n_steps:7.1f} ms  ({tDg * n_layers * n_steps / 1000:.3f} s)")

    # ---- Static memory footprint of expanded delta_M tensors ----
    bytes_per_float = 4
    print(f"\n  Expanded delta_M tensor size (static, fp32):")
    print(f"    dense N={N_dense}  6x6: {B*N_dense*36*bytes_per_float/1024**2:.1f} MB   "
          f"DxD: {B*N_dense*D*D*bytes_per_float/1024**2:.1f} MB")
    print(f"    fps   N={N_fps}   6x6: {B*N_fps*36*bytes_per_float/1024**2:.1f} MB   "
          f"DxD: {B*N_fps*D*D*bytes_per_float/1024**2:.1f} MB")


if __name__ == '__main__':
    main()
