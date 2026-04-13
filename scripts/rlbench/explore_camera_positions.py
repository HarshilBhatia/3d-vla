"""
Randomly sample camera positions around the robot workspace, render synthetic
views via point-cloud projection, and produce a self-contained HTML gallery
where you and collaborators can interactively shortlist N viable configs.

Supports PerAct zarrs ('pcd' + 'rgb' fields) and PerAct2 zarrs
('rgb' + 'depth' + 'extrinsics' + 'intrinsics' fields).

Usage (from 3d-vla/):
    python scripts/rlbench/explore_camera_positions.py \\
        --zarr  Peract_zarr/val.zarr \\
        --out   camera_explore/ \\
        --n_random  200 \\
        --n_frames  15 \\
        --n_shortlist 6

Output:
    <out>/index.html   -- interactive HTML gallery with shortlist selector
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import zarr
from scipy.spatial.transform import Rotation as ScipyR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from paths import ZARR_ROOT

# ── Constants ──────────────────────────────────────────────────────────────
PERACT_CAMERAS  = ["left_shoulder", "right_shoulder", "wrist", "front"]
PERACT2_CAMERAS = ["front", "wrist_left", "wrist_right"]


# ── Point cloud ────────────────────────────────────────────────────────────

def load_point_cloud(zarr_path, n_frames=15, seed=0):
    """
    Build a merged (N,3) world-XYZ + (N,3) uint8-RGB point cloud from a zarr.

    PerAct zarrs:  uses the 'pcd' field (per-pixel world XYZ).
    PerAct2 zarrs: back-projects 'depth' through stored 'extrinsics'/'intrinsics'.
    """
    z = zarr.open_group(zarr_path, mode="r")
    total  = z["rgb"].shape[0]
    rng    = np.random.default_rng(seed)
    frames = sorted(rng.choice(total, size=min(n_frames, total), replace=False).tolist())

    has_pcd = "pcd" in z
    has_ext = "extrinsics" in z and "intrinsics" in z and "depth" in z

    if not has_pcd and not has_ext:
        sys.exit("[ERROR] Zarr has neither 'pcd' (PerAct) nor 'extrinsics'+'depth' (PerAct2). "
                 "Cannot build a point cloud.")

    all_pts, all_cols = [], []

    for fi in frames:
        rgb_entry = z["rgb"][fi]               # (NCAM, 3, H, W) or (T, NCAM, ...)

        # Handle time-indexed rollouts
        if rgb_entry.ndim == 4 and rgb_entry.shape[0] > 6:
            t = 0
            rgb_entry = rgb_entry[t]

        ncam, _, H, W = rgb_entry.shape        # (NCAM, 3, H, W)

        if has_pcd:
            pcd_entry = z["pcd"][fi]
            if pcd_entry.ndim == 4 and pcd_entry.shape[0] > 6:
                pcd_entry = pcd_entry[0]

            for c in range(ncam):
                pts_c  = pcd_entry[c].astype(np.float32).reshape(3, -1).T  # (H*W, 3)
                cols_c = rgb_entry[c].reshape(3, -1).T                     # (H*W, 3) uint8
                valid  = np.isfinite(pts_c).all(1) & (pts_c[:, 2] > 0.005)
                idx    = np.where(valid)[0]
                if len(idx) == 0:
                    continue
                if len(idx) > 800:
                    idx = rng.choice(idx, 800, replace=False)
                all_pts.append(pts_c[idx])
                all_cols.append(cols_c[idx])

        else:   # PerAct2: back-project depth
            ext_entry   = z["extrinsics"][fi]
            intr_entry  = z["intrinsics"][fi]
            depth_entry = z["depth"][fi]
            if ext_entry.ndim == 3 and ext_entry.shape[0] > 6:
                ext_entry   = ext_entry[0]
                intr_entry  = intr_entry[0]
                depth_entry = depth_entry[0]

            for c in range(ncam):
                K_c  = intr_entry[c].astype(np.float64)
                E_c  = ext_entry[c].astype(np.float64)      # cam-to-world (4,4)
                d_c  = depth_entry[c].astype(np.float32)    # (H, W)
                valid_d = d_c > 0.01
                if valid_d.sum() < 10:
                    continue

                v_idx, u_idx = np.where(valid_d)
                Z    = d_c[v_idx, u_idx]
                X    = (u_idx - K_c[0, 2]) / K_c[0, 0] * Z
                Y    = (v_idx - K_c[1, 2]) / K_c[1, 1] * Z
                pts_cam = np.stack([X, Y, Z], axis=1)            # (M, 3)

                # Camera-to-world: X_world = R @ X_cam + t
                R_cw = E_c[:3, :3]
                t_cw = E_c[:3, 3]
                pts_w = pts_cam @ R_cw.T + t_cw                  # (M, 3)

                cols_c = rgb_entry[c, :, v_idx, u_idx].T         # (M, 3) uint8

                idx = np.arange(len(pts_w))
                if len(idx) > 800:
                    idx = rng.choice(idx, 800, replace=False)
                all_pts.append(pts_w[idx].astype(np.float32))
                all_cols.append(cols_c[idx])

    if not all_pts:
        sys.exit("[ERROR] No valid 3-D points found in zarr.")

    pts  = np.concatenate(all_pts,  axis=0)
    cols = np.concatenate(all_cols, axis=0)
    print("[INFO] Loaded {:,} world points from {} frames".format(len(pts), len(frames)))
    return pts, cols


def estimate_workspace(pts):
    """Return (centroid, radius, z_table) from the merged point cloud."""
    centroid = np.median(pts, axis=0)
    radius   = float(np.percentile(np.linalg.norm(pts - centroid, axis=1), 80))
    z_table  = float(np.percentile(pts[:, 2], 15))
    print("[INFO] Workspace centroid={}, radius={:.3f} m, table_z~{:.3f} m".format(
          centroid.round(3), radius, z_table))
    return centroid, radius, z_table


# ── Camera helpers ─────────────────────────────────────────────────────────

def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def look_at(eye, target, world_up=np.array([0., 0., 1.])):
    """
    Camera-to-world rotation matrix (3x3).
    Columns = camera X (right), Y (down-in-image), Z (forward) in world frame.
    Convention: image u increases rightward, v increases downward.
    """
    z = _normalize(np.asarray(target, float) - np.asarray(eye, float))
    if abs(float(np.dot(z, world_up))) > 0.99:
        world_up = np.array([0., 1., 0.])
    x = _normalize(np.cross(world_up, z))   # right
    y = np.cross(z, x)                      # right-handed: z × x = y  (det = +1)
    return np.column_stack([x, y, z])


def apply_roll(R, roll_deg):
    """Rotate around camera's own Z axis."""
    z_world = R[:, 2]
    return ScipyR.from_rotvec(z_world * np.deg2rad(roll_deg)).as_matrix() @ R


# ── Camera sampling ────────────────────────────────────────────────────────

def structured_cameras(center, radius, z_table):
    """Pre-defined sensible positions that mimic typical RLBench placements."""
    cx, cy, cz = center
    r = radius
    configs = [
        ("front",          [cx - r*1.8, cy,          cz + r*0.2]),
        ("front_high",     [cx - r*1.4, cy,          cz + r*0.9]),
        ("left_shoulder",  [cx - r*0.2, cy + r*1.5,  cz + r*1.3]),
        ("right_shoulder", [cx - r*0.2, cy - r*1.5,  cz + r*1.3]),
        ("overhead",       [cx,          cy,          cz + r*2.2]),
        ("left_side",      [cx + r*0.4,  cy + r*1.9,  cz + r*0.4]),
        ("right_side",     [cx + r*0.4,  cy - r*1.9,  cz + r*0.4]),
        ("back_high",      [cx + r*1.6,  cy,          cz + r*1.0]),
        ("front_low",      [cx - r*2.0, cy,          cz - r*0.2]),
        ("wrist_approx",   [cx + r*0.1, cy + r*0.3, cz + r*0.15]),
    ]
    cameras = []
    for name, pos_list in configs:
        pos = np.array(pos_list, dtype=float)
        pos[2] = max(pos[2], z_table + 0.05)
        cameras.append({
            "name": name, "pos": pos,
            "R": look_at(pos, center),
            "rz": 0.0, "structured": True,
        })
    return cameras


def sample_random_cameras(center, radius, z_table, n, seed, jitter_m=0.05, roll_deg=20.0):
    """
    Sample n cameras uniformly over a shell [r_min, r_max] around the workspace.
    All cameras look toward the workspace centre (+/- small jitter).
    """
    rng   = np.random.default_rng(seed)
    r_min = radius * 0.4
    r_max = radius * 2.8
    z_min = z_table + 0.05
    z_max = center[2] + radius * 3.0

    cameras, attempts = [], 0
    while len(cameras) < n and attempts < n * 30:
        attempts += 1

        # Uniform point in shell using rejection sampling from cube
        while True:
            dp   = rng.uniform(-r_max, r_max, 3)
            dist = np.linalg.norm(dp)
            if r_min <= dist <= r_max:
                break
        pos = center + dp
        if pos[2] < z_min or pos[2] > z_max:
            continue

        target = center + rng.normal(0, jitter_m, 3)
        roll   = float(rng.uniform(-roll_deg, roll_deg))
        R      = apply_roll(look_at(pos, target), roll)

        cameras.append({
            "name": "rand_{:04d}".format(len(cameras)),
            "pos": pos, "R": R, "rz": roll, "structured": False,
        })
    return cameras


# ── Rendering ──────────────────────────────────────────────────────────────

def make_K(fov_deg, img_size):
    f  = (img_size / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
    cx = cy = img_size / 2.0
    return np.array([[f, 0., cx], [0., f, cy], [0., 0., 1.]])


def render_view(pts, colors, cam_pos, cam_R, K, H=256, W=256):
    """
    Project (N,3) world points through a pinhole camera.
    cam_R: (3,3) camera-to-world rotation (columns = cam axes in world).
    Returns (H,W,3) uint8 image with 3×3 point splat.
    """
    # X_cam = R^T @ (X_world - O)  =>  (pts - cam_pos) @ cam_R  in batch form
    X_cam = (pts - cam_pos) @ cam_R             # (N, 3)

    valid = X_cam[:, 2] > 0.01
    X_cam = X_cam[valid]
    col   = colors[valid]

    if len(X_cam) == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    u = np.round(fx * X_cam[:, 0] / X_cam[:, 2] + cx).astype(np.int32)
    v = np.round(fy * X_cam[:, 1] / X_cam[:, 2] + cy).astype(np.int32)
    d = X_cam[:, 2]

    # Keep in-bounds (1px margin for splat)
    mask = (u >= 1) & (u < W - 1) & (v >= 1) & (v < H - 1)
    u, v, d, col = u[mask], v[mask], d[mask], col[mask]

    # Painter's algorithm: far-to-near
    order = np.argsort(-d)
    u, v, col = u[order], v[order], col[order]

    img = np.zeros((H, W, 3), dtype=np.uint8)
    for dv, du in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]:
        img[v + dv, u + du] = col

    return img


def coverage_score(img):
    return float((img.sum(axis=2) > 20).mean())


# ── PNG grid output ─────────────────────────────────────────────────────────

def write_png_grids(candidates, out_dir, title="Camera Positions", n_cols=5):
    """
    Write page_001.png, page_002.png, ... (20 candidates per page) and
    summary_structured.png (structured candidates only).

    Each cell shows the rendered view with its #IDX label so you can
    cross-reference against configs.json.
    """
    n_per_page = n_cols * 4
    n_pages    = max(1, (len(candidates) + n_per_page - 1) // n_per_page)
    saved      = []

    for p in range(n_pages):
        page = candidates[p * n_per_page : (p + 1) * n_per_page]
        n_rows = max(1, (len(page) + n_cols - 1) // n_cols)

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 2.6, n_rows * 3.0),
                                 facecolor="#0a0a14",
                                 squeeze=False)
        fig.subplots_adjust(wspace=0.04, hspace=0.45,
                            left=0.01, right=0.99, top=0.93, bottom=0.01)

        flat = axes.flat
        for j, cand in enumerate(page):
            ax   = next(flat)
            gidx = p * n_per_page + j
            img  = cand["render_img"]   # (H, W, 3) uint8

            ax.imshow(img)
            tag   = "[S]" if cand.get("structured") else ""
            label = "#{:03d} {} {}\n[{:.2f},{:.2f},{:.2f}]\ncov={:.0f}% d={:.2f}m".format(
                gidx, tag, cand["name"][:14],
                cand["pos"][0], cand["pos"][1], cand["pos"][2],
                cand.get("coverage", 0.0) * 100,
                cand.get("dist", 0.0))
            ax.set_title(label, fontsize=5.0, color="#a0c4ff", pad=2)
            ax.axis("off")
            color = "#44aaff" if cand.get("structured") else "#2a2a50"
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(color)
                spine.set_linewidth(1.5)

        for ax in flat:
            ax.axis("off")
            ax.set_facecolor("#0a0a14")

        fig.suptitle("{} — page {}/{} (#{}-#{})".format(
            title, p + 1, n_pages,
            p * n_per_page,
            min((p + 1) * n_per_page - 1, len(candidates) - 1)),
            color="white", fontsize=8, y=0.97)

        path = os.path.join(out_dir, "page_{:03d}.png".format(p + 1))
        fig.savefig(path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        saved.append(path)
        print("[PAGE] {}".format(path))

    # Summary: structured candidates only
    struct = [c for c in candidates if c.get("structured")]
    if struct:
        nc    = min(5, len(struct))
        nr    = max(1, (len(struct) + nc - 1) // nc)
        fig2, axes2 = plt.subplots(nr, nc,
                                   figsize=(nc * 3.0, nr * 3.5),
                                   facecolor="#0a0a14",
                                   squeeze=False)
        fig2.subplots_adjust(wspace=0.06, hspace=0.5,
                             left=0.01, right=0.99, top=0.90, bottom=0.01)
        flat2 = axes2.flat
        for j, cand in enumerate(struct):
            ax = next(flat2)
            ax.imshow(cand["render_img"])
            ax.set_title("{}\n[{:.2f},{:.2f},{:.2f}]\ncov={:.0f}% d={:.2f}m".format(
                cand["name"],
                cand["pos"][0], cand["pos"][1], cand["pos"][2],
                cand.get("coverage", 0.0) * 100,
                cand.get("dist", 0.0)),
                fontsize=7, color="#a0c4ff", pad=3)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#44aaff")
                spine.set_linewidth(2)
        for ax in flat2:
            ax.axis("off"); ax.set_facecolor("#0a0a14")

        fig2.suptitle("Structured camera positions", color="white",
                      fontsize=10, y=0.96)
        summary = os.path.join(out_dir, "summary_structured.png")
        fig2.savefig(summary, dpi=130, bbox_inches="tight",
                     facecolor=fig2.get_facecolor())
        plt.close(fig2)
        saved.insert(0, summary)
        print("[SUMMARY] {}".format(summary))

    return saved


def write_configs_json(candidates, out_dir):
    """Write configs.json: list of all candidates with their #IDX, name, pos, roll."""
    records = []
    for i, c in enumerate(candidates):
        records.append({
            "idx":              i,
            "name":             c["name"],
            "structured":       bool(c.get("structured", False)),
            "pos":              [round(float(v), 4) for v in c["pos"]],
            "roll_deg":         round(float(c.get("rz", 0.0)), 2),
            "coverage":         round(float(c.get("coverage", 0.0)), 4),
            "dist_from_center": round(float(c.get("dist", 0.0)), 4),
        })
    path = os.path.join(out_dir, "configs.json")
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    return path


# ── CLI & main ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Randomly sample camera positions and render novel views for shortlisting."
    )
    p.add_argument("--zarr",        default=None, help="Path to val.zarr (PerAct or PerAct2)")
    p.add_argument("--out",         default="camera_explore")
    p.add_argument("--n_random",    type=int,   default=200,
                   help="Number of random candidates (default: 200)")
    p.add_argument("--n_frames",    type=int,   default=15,
                   help="Frames to merge for point cloud (default: 15)")
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--fov_deg",     type=float, default=65.0,
                   help="FOV for rendered novel views in degrees (default: 65)")
    p.add_argument("--img_size",    type=int,   default=256)
    p.add_argument("--cols",        type=int,   default=5,
                   help="Columns per grid page (default: 5)")
    p.add_argument("--save_pngs",   action="store_true",
                   help="Also save individual view PNGs to <out>/candidate_XXXX/")
    return p.parse_args()


def main():
    args = parse_args()

    zarr_path = args.zarr or os.path.join(ZARR_ROOT, "val.zarr")
    if not zarr_path.endswith(".zarr"):
        zarr_path = zarr_path.rstrip("/") + ".zarr"
    if not os.path.isdir(zarr_path):
        sys.exit("[ERROR] Zarr not found: {}".format(zarr_path))

    os.makedirs(args.out, exist_ok=True)

    H = W = args.img_size
    K = make_K(args.fov_deg, args.img_size)
    print("[INFO] Rendering with FOV={:.0f}deg  f={:.1f}px  {}x{}".format(
          args.fov_deg, K[0, 0], H, W))

    # 1. Build point cloud
    pts, cols = load_point_cloud(zarr_path, args.n_frames, args.seed)

    # 2. Workspace geometry
    center, radius, z_table = estimate_workspace(pts)

    # 3. Camera list: structured + random
    cams  = structured_cameras(center, radius, z_table)
    cams += sample_random_cameras(center, radius, z_table, args.n_random, args.seed)
    print("[INFO] {} candidates ({} structured, {} random)".format(
          len(cams), sum(c["structured"] for c in cams), sum(not c["structured"] for c in cams)))

    # 4. Render each candidate
    candidates = []
    for i, cam in enumerate(cams):
        img = render_view(pts, cols, cam["pos"], cam["R"], K, H, W)
        cov = coverage_score(img)

        if args.save_pngs:
            png_dir = os.path.join(args.out, "candidate_{:04d}_{}".format(
                i, cam["name"].replace("/", "_")))
            os.makedirs(png_dir, exist_ok=True)
            try:
                from PIL import Image as PILImage
                PILImage.fromarray(img).save(os.path.join(png_dir, "view.png"))
            except ImportError:
                import imageio
                imageio.imwrite(os.path.join(png_dir, "view.png"), img)

        candidates.append({
            **cam,
            "render_img": img,
            "coverage":   cov,
            "dist":       float(np.linalg.norm(cam["pos"] - center)),
        })

        print("  [{:>4}/{}] {:<30s} cov={:.0f}%".format(
              i + 1, len(cams), cam["name"], cov * 100), end="\r", flush=True)

    print()  # newline

    # Sort: structured first, then by coverage desc
    candidates.sort(key=lambda c: (not c["structured"], -c["coverage"]))

    # 5. Write PNG grids + configs.json
    pages = write_png_grids(candidates, args.out,
                            title="PerAct Camera Positions",
                            n_cols=args.cols)
    cfgp  = write_configs_json(candidates, args.out)

    print("\n[DONE] {} candidates".format(len(candidates)))
    print("  Browse : {}  (page_001.png … page_{:03d}.png)".format(
          args.out, len([p for p in pages if "page_" in p])))
    print("  Summary: {}".format(os.path.join(args.out, "summary_structured.png")))
    print("  Index  : {}".format(cfgp))

    # Print coverage summary to stdout
    covered = [c for c in candidates if c["coverage"] > 0.15]
    print("\n[SUMMARY] {} / {} candidates have coverage > 15%".format(
          len(covered), len(candidates)))
    print("  Top 10 by coverage:")
    for c in sorted(candidates, key=lambda x: -x["coverage"])[:10]:
        print("    {:<30s} cov={:.0f}%  dist={:.2f}m  pos=[{:.2f},{:.2f},{:.2f}]".format(
              c["name"], c["coverage"] * 100, c["dist"],
              c["pos"][0], c["pos"][1], c["pos"][2]))


if __name__ == "__main__":
    main()
