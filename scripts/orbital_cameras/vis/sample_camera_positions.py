"""
Sample and visualize candidate camera positions for PerAct2.

Loads one frame from a PerAct2 zarr dataset, applies a grid of global
camera-rig perturbations (rotations around Z, X/Y/Z translations), renders
the resulting camera views alongside 3D position plots, and writes a
self-contained HTML summary that you can share with collaborators.

Camera-to-world extrinsics are transformed as:  E_new = T_global @ E
where T_global encodes rotation (around the specified axis) + translation.
When rotation is applied, the RGB image is counter-rotated in-plane so the
stored view stays consistent with the new extrinsics.

Usage (from 3d-vla/):
    python scripts/rlbench/sample_camera_positions.py \\
        --zarr Peract2_zarr/bimanual_lift_tray/val.zarr \\
        --out camera_samples/ \\
        --rotate_z "-30,-15,0,15,30" \\
        --translate_x "-0.2,0,0.2" \\
        --translate_z "-0.1,0,0.1" \\
        --rollout 0 --frame 0

    # Custom candidates (overrides grid sweep):
    python scripts/rlbench/sample_camera_positions.py \\
        --zarr Peract2_zarr/bimanual_lift_tray/val.zarr \\
        --out camera_samples/ \\
        --candidates "rz=0,tx=0,ty=0,tz=0;rz=15,tx=0.1,ty=0,tz=0.05"

Outputs:
    <out>/candidate_<id>/views.png   -- 3 camera views + 3D camera plot  (with --save_pngs)
    <out>/index.html                 -- shareable self-contained HTML
"""

import argparse
import base64
import io
import os
import sys
from itertools import product
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import numpy as np
import zarr
from scipy.ndimage import rotate as ndimage_rotate
from scipy.spatial.transform import Rotation as ScipyR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
ZARR_ROOT = "Peract2_zarr"

CAMERAS = ["front", "wrist_left", "wrist_right"]
CAMERA_COLORS = ["tab:blue", "tab:orange", "tab:green"]

# Camera names for the original PerAct 4-camera format
PERACT_CAMERAS = ["left_shoulder", "right_shoulder", "wrist", "front"]
PERACT_CAMERA_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


# ---------------------------------------------------------------------------
# Extrinsics estimation from per-pixel world coordinates (for PerAct zarr)
# ---------------------------------------------------------------------------

def estimate_extrinsics_from_pcd(pcd_chw: np.ndarray) -> np.ndarray:
    """Estimate camera-to-world extrinsics via DLT from a per-pixel XYZ depth map.

    Parameters
    ----------
    pcd_chw : np.ndarray
        Shape (3, H, W) with world-space XYZ at each pixel (float, 0 = invalid).

    Returns
    -------
    np.ndarray
        4×4 camera-to-world homogeneous matrix.  Falls back to eye(4) on failure.
    """
    C, H, W = pcd_chw.shape
    p = pcd_chw.astype(np.float32)

    # Sample a 32×32 uniform grid of pixels
    ys = np.linspace(0, H - 1, 32, dtype=int)
    xs = np.linspace(0, W - 1, 32, dtype=int)
    yy, xx = np.meshgrid(ys, xs)
    u_all = xx.flatten()
    v_all = yy.flatten()

    X_all = p[0, v_all, u_all]
    Y_all = p[1, v_all, u_all]
    Z_all = p[2, v_all, u_all]

    # Keep only valid (non-zero, finite) points
    valid = (
        np.isfinite(X_all) & np.isfinite(Y_all) & np.isfinite(Z_all)
        & ((np.abs(X_all) + np.abs(Y_all) + np.abs(Z_all)) > 0)
    )
    u, v = u_all[valid].astype(np.float64), v_all[valid].astype(np.float64)
    X, Y, Z = X_all[valid].astype(np.float64), Y_all[valid].astype(np.float64), Z_all[valid].astype(np.float64)

    if len(u) < 12:
        print("[WARN] Too few valid PCD points for extrinsics estimation; using identity.")
        return np.eye(4)

    # Direct Linear Transform (DLT): solve for 3×4 camera matrix P
    # such that s * [u, v, 1]^T = P * [X, Y, Z, 1]^T
    N = len(u)
    A = np.zeros((2 * N, 12), dtype=np.float64)
    ones = np.ones(N)
    zeros = np.zeros(N)
    A[0::2,  :4] = np.column_stack([X, Y, Z, ones])
    A[0::2, 8:] = np.column_stack([-u * X, -u * Y, -u * Z, -u])
    A[1::2, 4:8] = np.column_stack([X, Y, Z, ones])
    A[1::2, 8:] = np.column_stack([-v * X, -v * Y, -v * Z, -v])

    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        print("[WARN] SVD failed for extrinsics estimation; using identity.")
        return np.eye(4)

    P = Vt[-1].reshape(3, 4)   # 3×4 projection matrix (up to scale)

    # Camera centre: null-space of P
    _, _, Vt2 = np.linalg.svd(P)
    ch = Vt2[-1]
    if abs(ch[3]) < 1e-8:
        print("[WARN] Degenerate camera centre; using identity.")
        return np.eye(4)
    cam_center = ch[:3] / ch[3]

    # RQ-decompose P[:, :3] → K * R  (K upper-triangular, R rotation)
    from scipy.linalg import rq
    K_est, R_est = rq(P[:, :3])

    # Enforce positive diagonal on K
    D = np.diag(np.sign(np.diag(K_est)))
    R_est = D @ R_est

    # Build 4×4 camera-to-world: rotation part = R_est^T, translation = cam_center
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_est.T
    T[:3, 3] = cam_center
    return T


# ---------------------------------------------------------------------------
# Transform helpers (mirrors peract2_to_zarr.py)
# ---------------------------------------------------------------------------

def build_transform(rotate_deg, axis, tx, ty, tz):
    """Build a 4x4 rigid-body transform: rotation around axis then translation."""
    T = np.eye(4, dtype=np.float64)
    if rotate_deg != 0.0:
        T[:3, :3] = ScipyR.from_euler(axis, np.deg2rad(rotate_deg)).as_matrix()
    T[:3, 3] = [tx, ty, tz]
    return T


def apply_transform_to_extrinsics(E, T):
    """E: (NCAM, 4, 4) camera-to-world.  Returns E_new = T @ E, same shape."""
    return np.einsum("ij,cjk->cik", T, E.astype(np.float64))


def rotate_rgb(img_chw, angle_deg):
    """Rotate a (3, H, W) uint8 image in-plane by angle_deg degrees."""
    if angle_deg == 0.0:
        return img_chw
    out = np.empty_like(img_chw)
    for ch in range(3):
        out[ch] = ndimage_rotate(
            img_chw[ch], angle_deg, axes=(0, 1),
            reshape=False, order=1, mode="constant", cval=0
        )
    return out


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def camera_axes_from_extrinsics(E, length=0.05):
    """
    Return origin positions and XYZ axis endpoints for each camera.
    E: (NCAM, 4, 4) camera-to-world.
    """
    origins = E[:, :3, 3]          # (NCAM, 3)
    axis_dirs = E[:, :3, :3]       # (NCAM, 3, 3) columns = cam X/Y/Z in world
    axes = np.stack(
        [
            np.stack([origins, origins + axis_dirs[:, :, i] * length], axis=1)
            for i in range(3)
        ],
        axis=1,
    )   # (NCAM, 3, 2, 3)
    return origins, axes


def make_candidate_figure(rgb_ncam_chw, extrinsics, baseline_extrinsics, title, camera_names,
                          camera_colors=None):
    """
    Build a matplotlib figure: camera views on the left, 3D plot on the right.

    rgb_ncam_chw: (NCAM, 3, H, W) uint8
    extrinsics:   (NCAM, 4, 4)
    baseline_extrinsics: (NCAM, 4, 4) or None
    """
    if camera_colors is None:
        camera_colors = CAMERA_COLORS
    ncam = len(camera_names)
    fig = plt.figure(figsize=(5 * (ncam + 1), 5), facecolor="#1a1a2e")
    gs = gridspec.GridSpec(
        1, ncam + 1,
        figure=fig, wspace=0.05,
        left=0.02, right=0.98, top=0.88, bottom=0.05
    )

    axis_colors = ["red", "limegreen", "dodgerblue"]

    # Camera image panels
    for i, (name, color) in enumerate(zip(camera_names, CAMERA_COLORS)):
        ax = fig.add_subplot(gs[0, i])
        img = np.transpose(rgb_ncam_chw[i], (1, 2, 0))
        ax.imshow(img)
        ax.set_title(name, color=color, fontsize=11, fontweight="bold", pad=4)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    # 3D camera position plot
    ax3d = fig.add_subplot(gs[0, ncam], projection="3d")
    ax3d.set_facecolor("#0d0d1a")
    ax3d.grid(True, color="#333355", linewidth=0.5)

    origins, axes = camera_axes_from_extrinsics(extrinsics)
    for i, (name, color) in enumerate(zip(camera_names, CAMERA_COLORS)):
        ax3d.scatter(*origins[i], s=80, color=color, zorder=5, label=name)
        for j in range(3):
            seg = axes[i, j]    # (2, 3)
            ax3d.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                      color=axis_colors[j], linewidth=1.2, alpha=0.8)

    if baseline_extrinsics is not None:
        base_origins = baseline_extrinsics[:, :3, 3]
        for i in range(ncam):
            ax3d.scatter(*base_origins[i], s=30, color=CAMERA_COLORS[i],
                         marker="x", alpha=0.35, zorder=3)

    ax3d.legend(fontsize=7, loc="upper left", facecolor="#0d0d1a",
                labelcolor="white", framealpha=0.6)
    ax3d.set_xlabel("X", color="white", fontsize=8)
    ax3d.set_ylabel("Y", color="white", fontsize=8)
    ax3d.set_zlabel("Z", color="white", fontsize=8)
    ax3d.tick_params(colors="white", labelsize=7)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False

    fig.suptitle(title, color="white", fontsize=13, fontweight="bold", y=0.97)
    return fig


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

HTML_STYLE = """
<style>
  body {{ background: #0f0f1e; color: #e0e0f0;
         font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; }}
  h1   {{ text-align: center; color: #a0c4ff; margin-bottom: 4px; }}
  .subtitle {{ text-align: center; color: #7070a0; margin-bottom: 30px; font-size: 0.9em; }}
  .grid {{ display: grid;
          grid-template-columns: repeat(auto-fill, minmax(560px, 1fr));
          gap: 18px; }}
  .card {{ background: #1a1a2e; border: 1px solid #2a2a4a;
          border-radius: 8px; padding: 12px; }}
  .card h3  {{ margin: 0 0 8px 0; color: #a0c4ff; font-size: 0.95em; }}
  .card img {{ width: 100%; border-radius: 4px; }}
  .params   {{ font-size: 0.78em; color: #8080b0; margin-top: 6px; font-family: monospace; }}
  .baseline {{ border: 2px solid #3a7aff; }}
</style>
"""


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def write_html(candidates, out_dir, zarr_path):
    cards = []
    for c in candidates:
        css_class = "card baseline" if c["is_baseline"] else "card"
        params_str = (
            "rz={:+.1f}deg  tx={:+.3f}m  ty={:+.3f}m  tz={:+.3f}m".format(
                c["rz"], c["tx"], c["ty"], c["tz"]
            )
        )
        cards.append(
            '<div class="{}">'
            '<h3>{}</h3>'
            '<img src="data:image/png;base64,{}" alt="{}">'
            '<div class="params">{}</div>'
            '</div>'.format(
                css_class, c["label"], c["b64"], c["label"], params_str
            )
        )

    cards_html = "\n".join(cards)
    html = (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        "<meta charset='UTF-8'>\n"
        "<title>PerAct2 Camera Position Samples</title>\n"
        + HTML_STYLE +
        "</head>\n<body>\n"
        "<h1>PerAct2 Camera Position Samples</h1>\n"
        "<div class='subtitle'>"
        "Zarr: <code>{}</code> &nbsp;&middot;&nbsp; {} candidates"
        " &nbsp;&middot;&nbsp; Blue-bordered = baseline (no perturbation)"
        "</div>\n"
        "<div class='grid'>\n{}\n</div>\n"
        "</body>\n</html>"
    ).format(zarr_path, len(candidates), cards_html)

    path = os.path.join(out_dir, "index.html")
    with open(path, "w") as f:
        f.write(html)
    return path


# ---------------------------------------------------------------------------
# Candidate parsing
# ---------------------------------------------------------------------------

def parse_floats(s):
    return [float(v.strip()) for v in s.split(",") if v.strip()]


def parse_candidates(spec):
    """Parse semicolon-separated 'rz=V,tx=V,ty=V,tz=V' strings."""
    result = []
    for entry in spec.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        d = {"rz": 0.0, "tx": 0.0, "ty": 0.0, "tz": 0.0}
        for kv in entry.split(","):
            k, v = kv.strip().split("=")
            d[k.strip()] = float(v.strip())
        result.append(d)
    return result


def build_grid_candidates(rotate_z, translate_x, translate_y, translate_z):
    candidates = []
    for rz, tx, ty, tz in product(rotate_z, translate_x, translate_y, translate_z):
        candidates.append({"rz": rz, "tx": tx, "ty": ty, "tz": tz})
    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Sample PerAct2 camera positions and write a shareable HTML report."
    )
    p.add_argument("--zarr", default=None,
                   help="Path to a PerAct2 val.zarr (default: ZARR_ROOT/val.zarr)")
    p.add_argument("--out", default="camera_samples",
                   help="Output directory (default: camera_samples/)")
    p.add_argument("--rollout", type=int, default=0,
                   help="Which rollout/frame index to use from the zarr (0-indexed)")
    p.add_argument("--frame", type=int, default=0,
                   help="Keyframe index within the rollout (only used for rollout-indexed zarrs)")
    p.add_argument("--rotate_axis", default="z", choices=["x", "y", "z"],
                   help="Rotation axis for the global rig transform (default: z)")

    # Grid sweep parameters
    p.add_argument("--rotate_z", default="0",
                   help="Comma-separated rotation values in degrees (e.g. '-30,-15,0,15,30')")
    p.add_argument("--translate_x", default="0",
                   help="Comma-separated X translations in metres (e.g. '-0.2,0,0.2')")
    p.add_argument("--translate_y", default="0",
                   help="Comma-separated Y translations in metres")
    p.add_argument("--translate_z", default="0",
                   help="Comma-separated Z translations in metres (e.g. '-0.1,0,0.1')")

    # Alternative: explicit candidates
    p.add_argument("--candidates", default=None,
                   help=(
                       "Explicit semicolon-separated candidates: "
                       "'rz=10,tx=0,ty=0,tz=0;rz=-10,tx=0,ty=0,tz=0.1' "
                       "(overrides grid sweep flags)"
                   ))

    p.add_argument("--save_pngs", action="store_true",
                   help="Also save individual PNGs to <out>/candidate_XXX/views.png")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Locate zarr ---
    zarr_path = args.zarr or os.path.join(ZARR_ROOT, "val.zarr")
    if not zarr_path.endswith(".zarr"):
        zarr_path = zarr_path.rstrip("/") + ".zarr"
    if not os.path.isdir(zarr_path):
        sys.exit("[ERROR] Zarr not found: {}".format(zarr_path))

    group = zarr.open_group(zarr_path, mode="r")
    if "rgb" not in group:
        sys.exit("[ERROR] 'rgb' not found in zarr. Keys: {}".format(list(group.keys())))

    # ------------------------------------------------------------------
    # Determine zarr format:
    #   Peract2 format: has 'extrinsics' key  (ground-truth per-frame camera poses)
    #   Peract  format: has 'pcd' key instead (per-pixel world XYZ; estimate extrinsics)
    # ------------------------------------------------------------------
    has_extrinsics = "extrinsics" in group
    has_pcd        = "pcd" in group

    if not has_extrinsics and not has_pcd:
        sys.exit(
            "[ERROR] Zarr has neither 'extrinsics' nor 'pcd'. Keys: {}".format(
                list(group.keys())
            )
        )

    if not has_extrinsics:
        print(
            "[INFO] No 'extrinsics' key found — falling back to PerAct format.\n"
            "       Camera poses will be estimated from 'pcd' (per-pixel world XYZ).\n"
            "       Perturbation simulation is not available in this mode; "
            "only the baseline (original) views will be shown."
        )

    # rgb shape: (N, NCAM, 3, H, W)
    rgb_arr = group["rgb"]
    total   = rgb_arr.shape[0]

    rollout_idx = args.rollout
    if rollout_idx >= total:
        sys.exit("[ERROR] --rollout {} out of range (zarr has {} entries)".format(rollout_idx, total))

    rgb_entry = rgb_arr[rollout_idx]  # (NCAM, 3, H, W)  or  (T, NCAM, 3, H, W)

    # Handle rollout-indexed zarrs (Peract2 time-indexed)
    if has_extrinsics:
        ext_arr   = group["extrinsics"]
        ext_entry = ext_arr[rollout_idx].astype(np.float64)

        if rgb_entry.ndim == 4 and rgb_entry.shape[1] == 3 and rgb_entry.shape[0] > 6:
            frame_idx = args.frame
            if frame_idx >= rgb_entry.shape[0]:
                sys.exit("[ERROR] --frame {} out of range (rollout has {} keyframes)".format(
                    frame_idx, rgb_entry.shape[0]))
            rgb_frame = rgb_entry[frame_idx]
            ext_frame = ext_entry[frame_idx].astype(np.float64)
            print("[INFO] rollout-indexed zarr: using rollout={}, frame={}".format(rollout_idx, frame_idx))
        else:
            rgb_frame = rgb_entry
            ext_frame = ext_entry
            print("[INFO] frame-indexed zarr: using index={}".format(rollout_idx))
    else:
        # PerAct format: estimate extrinsics from PCD
        rgb_frame = rgb_entry  # (NCAM, 3, H, W)
        pcd_entry = group["pcd"][rollout_idx]  # (NCAM, 3, H, W)
        ncam_pcd  = pcd_entry.shape[0]
        print("[INFO] Estimating extrinsics from PCD for {} cameras…".format(ncam_pcd))
        ext_frame = np.stack([
            estimate_extrinsics_from_pcd(pcd_entry[c].astype(np.float32))
            for c in range(ncam_pcd)
        ])  # (NCAM, 4, 4)
        print("[INFO] frame-indexed PerAct zarr: using index={}".format(rollout_idx))

    print("[INFO] rgb shape: {}  extrinsics shape: {}".format(rgb_frame.shape, ext_frame.shape))

    # Validate NCAM and pick camera names / colours
    ncam = rgb_frame.shape[0]
    if has_extrinsics:
        camera_names  = CAMERAS[:ncam]
        camera_colors = CAMERA_COLORS[:ncam]
    else:
        camera_names  = PERACT_CAMERAS[:ncam]
        camera_colors = PERACT_CAMERA_COLORS[:ncam]

    baseline_extrinsics = ext_frame.copy()

    # --- Build candidate list ---
    if not has_extrinsics:
        # PerAct mode: only show baseline (can't re-render from new positions)
        candidates = [{"rz": 0.0, "tx": 0.0, "ty": 0.0, "tz": 0.0}]
        print("[INFO] PerAct mode: showing baseline views only (perturbation not supported without ground-truth extrinsics)")
    elif args.candidates:
        candidates = parse_candidates(args.candidates)
        print("[INFO] Using {} explicit candidates".format(len(candidates)))
    else:
        rz_vals = parse_floats(args.rotate_z)
        tx_vals = parse_floats(args.translate_x)
        ty_vals = parse_floats(args.translate_y)
        tz_vals = parse_floats(args.translate_z)
        candidates = build_grid_candidates(rz_vals, tx_vals, ty_vals, tz_vals)
        print("[INFO] Grid: {}rz x {}tx x {}ty x {}tz = {} candidates".format(
            len(rz_vals), len(tx_vals), len(ty_vals), len(tz_vals), len(candidates)))

    # Ensure baseline is present (for Peract2 mode)
    if has_extrinsics:
        has_baseline = any(
            c["rz"] == 0.0 and c["tx"] == 0.0 and c["ty"] == 0.0 and c["tz"] == 0.0
            for c in candidates
        )
        if not has_baseline:
            candidates.insert(0, {"rz": 0.0, "tx": 0.0, "ty": 0.0, "tz": 0.0})
            print("[INFO] Prepended baseline (no perturbation) to candidate list")

    os.makedirs(args.out, exist_ok=True)

    # --- Process each candidate ---
    html_candidates = []
    for idx, cand in enumerate(candidates):
        rz, tx, ty, tz = cand["rz"], cand["tx"], cand["ty"], cand["tz"]
        is_baseline = (rz == 0.0 and tx == 0.0 and ty == 0.0 and tz == 0.0)

        T = build_transform(rz, args.rotate_axis, tx, ty, tz)
        new_ext = apply_transform_to_extrinsics(ext_frame, T)

        new_rgb = np.empty_like(rgb_frame)
        for c_idx in range(ncam):
            new_rgb[c_idx] = rotate_rgb(rgb_frame[c_idx], -rz)

        if is_baseline:
            label = "Baseline (original)"
        else:
            label = "#{:03d}  rz={:+.1f}deg  tx={:+.3f}  ty={:+.3f}  tz={:+.3f}".format(
                idx, rz, tx, ty, tz)

        fig = make_candidate_figure(
            rgb_ncam_chw=new_rgb,
            extrinsics=new_ext,
            baseline_extrinsics=baseline_extrinsics if not is_baseline else None,
            title=label,
            camera_names=camera_names,
            camera_colors=camera_colors,
        )

        b64 = fig_to_base64(fig)

        if args.save_pngs:
            png_dir = os.path.join(args.out, "candidate_{:03d}".format(idx))
            os.makedirs(png_dir, exist_ok=True)
            png_path = os.path.join(png_dir, "views.png")
            fig.savefig(png_path, dpi=120, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print("  [SAVE] {}".format(png_path))

        plt.close(fig)

        html_candidates.append({
            "label": label, "b64": b64,
            "rz": rz, "tx": tx, "ty": ty, "tz": tz,
            "is_baseline": is_baseline,
        })
        print("  [{}/{}] {}".format(idx + 1, len(candidates), label))

    html_path = write_html(html_candidates, args.out, zarr_path)
    print("\n[DONE] {} candidates written to:  {}".format(len(candidates), html_path))
    print("       Open in a browser and share with collaborators.")


if __name__ == "__main__":
    main()
