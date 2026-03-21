"""
visualise_eef_projection.py

Projects EEF position from robot base frame into the ext1 and ext2 camera image
planes and saves annotated mp4(s) — a direct sanity check for extrinsic correctness.

If the extrinsics are correct, the red dot should track the robot gripper and the
colored depth points from one camera should land on the correct pixels in the other.

Layout (2×2 per video):
  | ext1 RGB + colored ext2-depth pts projected in + EEF | ext2 RGB + colored ext1-depth pts + EEF |
  | ext1 depth map (TURBO)                               | ext2 depth map (TURBO)                  |

Output modes:
  No --cam2cam-json  →  one video at --output (uses metadata extrinsics)
  With --cam2cam-json → two videos:
      {stem}_metadata.mp4  (metadata extrinsics)
      {stem}_cam2cam.mp4   (cam2cam annotation extrinsics)

Usage:
    # metadata only
    python data_processing/visualise_eef_projection.py \
        --dataset-dir /work/nvme/bgkz/droid_raw_large_superset \
        --raw-dir /work/nvme/bgkz/droid_multilab_raw \
        --depth-dir /work/nvme/bgkz/droid_multilab_depths \
        --canonical-id ILIAD+50aee79f+2023-07-12-21h-13m-44s \
        --output eef_projection.mp4

    # metadata + cam2cam annotation comparison
    python data_processing/visualise_eef_projection.py \
        --dataset-dir /work/nvme/bgkz/droid_raw_large_superset \
        --raw-dir /work/nvme/bgkz/droid_multilab_raw \
        --depth-dir /work/nvme/bgkz/droid_multilab_depths \
        --canonical-id ILIAD+50aee79f+2023-07-12-21h-13m-44s \
        --cam2cam-json /work/nvme/bgkz/droid_annotations/cam2cam_extrinsics.json \
        --output eef_projection.mp4
"""

import argparse
import json
from pathlib import Path

import av
import blosc
import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

_IMG_H = 180
_IMG_W = 320
_NATIVE_W = 1280
_NATIVE_H = 720
_SCALE_X = _IMG_W / _NATIVE_W   # 0.25
_SCALE_Y = _IMG_H / _NATIVE_H   # 0.25


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_meta(raw_dir: Path, canonical_id: str) -> dict:
    ep_dir = raw_dir / canonical_id
    meta_files = list(ep_dir.glob("metadata_*.json"))
    if not meta_files:
        raise ValueError(f"No metadata_*.json in {ep_dir}")
    return json.loads(meta_files[0].read_text())


def dof_to_mat(dof) -> np.ndarray:
    """[tx, ty, tz, rx, ry, rz] → 4×4 cam→base. Rotation: Euler XYZ."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler("xyz", dof[3:]).as_matrix()
    T[:3, 3] = dof[:3]
    return T


def load_cam2cam(cam2cam_json: Path, cid: str):
    """
    Load extrinsics + intrinsics from cam2cam_extrinsics.json annotations.

    Returns (T1, T2, intr1, intr2) where:
      T1/T2   are 4×4 cam→base pose matrices (left_cam=ext1, right_cam=ext2)
      intr1/2 are [fx, fy, cx, cy] scaled to _IMG_H × _IMG_W
    """
    with open(cam2cam_json) as f:
        cam2cam = json.load(f)
    if cid not in cam2cam:
        raise ValueError(f"{cid} not found in {cam2cam_json}")
    ep = cam2cam[cid]
    if "left_cam" not in ep or "right_cam" not in ep:
        raise ValueError(f"Missing left_cam/right_cam for {cid} in {cam2cam_json}")

    T1 = np.array(ep["left_cam"]["pose"],  dtype=np.float64)
    T2 = np.array(ep["right_cam"]["pose"], dtype=np.float64)
    if T1.shape != (4, 4) or T2.shape != (4, 4):
        raise ValueError(f"Expected 4×4 pose matrices for {cid}, got {T1.shape}/{T2.shape}")

    def _intr(c):
        f = c["focal"]
        cx, cy = c["principal_point"]
        return np.array([f * _SCALE_X, f * _SCALE_Y, cx * _SCALE_X, cy * _SCALE_Y])

    return T1, T2, _intr(ep["left_cam"]), _intr(ep["right_cam"])


def load_eef_positions(raw_dir: Path, canonical_id: str) -> np.ndarray:
    """(T, 3) EEF xyz in robot base frame."""
    traj = raw_dir / canonical_id / "trajectory.h5"
    with h5py.File(traj, "r") as f:
        return f["observation/robot_state/cartesian_position"][:, :3].astype(np.float64)


def decode_all_frames(mp4_path: Path) -> list[np.ndarray]:
    frames = []
    with av.open(str(mp4_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for packet in container.demux(stream):
            for frame in packet.decode():
                frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def load_depth(depth_dir: Path, cid: str, serial: str):
    """Load full depth array (T, H, W) float32, or None if not present."""
    shape_p = depth_dir / cid / serial / "shape.npy"
    blosc_p = depth_dir / cid / serial / "depth.blosc"
    if not shape_p.exists() or not blosc_p.exists():
        return None
    shape = np.load(shape_p)
    raw   = blosc_p.read_bytes()
    return np.frombuffer(blosc.decompress(raw), dtype=np.float32).reshape(shape)


# ── Depth unprojection + reprojection ────────────────────────────────────────

def unproject_depth(depth_frame, intrinsics, T_cam2base, rgb_frame, max_pts=2000):
    """
    Unproject depth frame → world-frame (N,3) points with (N,3) uint8 RGB colors,
    subsampled to max_pts.  Returns (pts_world, colors).
    """
    depth = cv2.resize(depth_frame, (_IMG_W, _IMG_H), interpolation=cv2.INTER_NEAREST)
    fx, fy, cx, cy = intrinsics
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 3.0)
    ys, xs = np.where(valid)
    D = depth[ys, xs]
    pts_cam = np.stack([(xs - cx) * D / fx, (ys - cy) * D / fy, D, np.ones_like(D)], axis=-1)
    pts_world = (pts_cam @ T_cam2base.T)[:, :3]

    rgb = cv2.resize(rgb_frame, (_IMG_W, _IMG_H))
    colors = rgb[ys, xs]  # (N, 3) uint8

    if len(pts_world) > max_pts:
        idx = np.random.choice(len(pts_world), max_pts, replace=False)
        pts_world = pts_world[idx]
        colors    = colors[idx]
    return pts_world, colors



def draw_projected_pts(bgr, pts_world, colors, T_cam2base, intrinsics):
    """Unproject and draw colored depth dots onto bgr in-place."""
    if len(pts_world) == 0:
        return
    T_base2cam = np.linalg.inv(T_cam2base)
    hom = np.hstack([pts_world, np.ones((len(pts_world), 1))])
    pts_cam = (hom @ T_base2cam.T)[:, :3]

    valid = pts_cam[:, 2] > 0
    pts_cam = pts_cam[valid]
    cols    = colors[valid]
    if len(pts_cam) == 0:
        return

    fx, fy, cx, cy = intrinsics
    us = (fx * pts_cam[:, 0] / pts_cam[:, 2] + cx).astype(int)
    vs = (fy * pts_cam[:, 1] / pts_cam[:, 2] + cy).astype(int)

    in_frame = (us >= 0) & (us < _IMG_W) & (vs >= 0) & (vs < _IMG_H)
    us, vs, cols = us[in_frame], vs[in_frame], cols[in_frame]
    for u, v, c in zip(us, vs, cols):
        cv2.circle(bgr, (int(u), int(v)), 2, (int(c[2]), int(c[1]), int(c[0])), -1)


# ── Visualisation helpers ─────────────────────────────────────────────────────

def depth_to_bgr(depth_frame, vmax=2.0) -> np.ndarray:
    """
    Colorize a single depth frame (float32, metres) to BGR for display.
    Uses TURBO colormap. Invalid/zero depths shown as dark grey background.
    """
    depth = cv2.resize(depth_frame, (_IMG_W, _IMG_H), interpolation=cv2.INTER_NEAREST)
    valid = np.isfinite(depth) & (depth > 0)
    normed = np.zeros((_IMG_H, _IMG_W), dtype=np.uint8)
    normed[valid] = np.clip(depth[valid] / vmax, 0.0, 1.0) * 255
    bgr = np.full((_IMG_H, _IMG_W, 3), 40, dtype=np.uint8)
    colored = cv2.applyColorMap(normed, cv2.COLORMAP_TURBO)
    bgr[valid] = colored[valid]
    return bgr


def project_eef(eef_xyz, T_cam2base, intrinsics):
    """
    Returns (u, v) pixel coords of EEF in camera image, or None if behind camera.
    intrinsics: [fx, fy, cx, cy] at _IMG_H × _IMG_W.
    """
    T_base2cam = np.linalg.inv(T_cam2base)
    eef_hom = np.array([eef_xyz[0], eef_xyz[1], eef_xyz[2], 1.0])
    eef_cam = T_base2cam @ eef_hom
    if eef_cam[2] <= 0:
        return None
    fx, fy, cx, cy = intrinsics
    u = fx * eef_cam[0] / eef_cam[2] + cx
    v = fy * eef_cam[1] / eef_cam[2] + cy
    return float(u), float(v)


def draw_eef(bgr, uv, label=None):
    """Draw EEF dot (red with white ring) onto bgr in-place."""
    if uv is None:
        return
    u, v = int(round(uv[0])), int(round(uv[1]))
    cv2.circle(bgr, (u, v), 8, (0, 0, 255), -1)
    cv2.circle(bgr, (u, v), 8, (255, 255, 255), 2)
    if label:
        cv2.putText(bgr, label, (u + 10, v + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)


def render_frame(bgr1, bgr2, depth_bgr1, depth_bgr2,
                 d1, d2, img1_rgb, img2_rgb,
                 eef, T1, T2, intr1, intr2,
                 T1_meta, T2_meta, intr1_meta, intr2_meta,
                 source_label, depth_vmax, frame_idx) -> np.ndarray:
    """
    Build one 2×2 output frame.
      top row:    RGB + cross-projected colored depth dots + EEF dot
      bottom row: depth maps (TURBO)

    T1/T2, intr1/intr2:               extrinsics for cross-depth projection (metadata or cam2cam)
    T1_meta/T2_meta, intr1/2_meta:    metadata cam→base extrinsics, always used for EEF dot
                                       (cam2cam poses are NOT in robot base frame)
    d1/d2:        raw float32 depth frames for ext1/ext2 (or None)
    img1/2_rgb:   uint8 RGB frames for coloring projected points
    """
    panel1 = bgr1.copy()
    panel2 = bgr2.copy()

    # Cross-project: ext2 depth → ext1 image, and ext1 depth → ext2 image
    if d2 is not None:
        pts2, cols2 = unproject_depth(d2, intr2, T2, img2_rgb)
        draw_projected_pts(panel1, pts2, cols2, T1, intr1)
    if d1 is not None:
        pts1, cols1 = unproject_depth(d1, intr1, T1, img1_rgb)
        draw_projected_pts(panel2, pts1, cols1, T2, intr2)

    # EEF dot always uses metadata cam→base extrinsics (EEF is in robot base frame)
    draw_eef(panel1, project_eef(eef, T1_meta, intr1_meta))
    draw_eef(panel2, project_eef(eef, T2_meta, intr2_meta))

    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1
    cv2.putText(panel1,     f"ext1 | ext2-depth pts | [{source_label}] f={frame_idx}", (5, 15), font, fs, (255,255,255), th)
    cv2.putText(panel2,     f"ext2 | ext1-depth pts | [{source_label}]",               (5, 15), font, fs, (255,255,255), th)
    cv2.putText(depth_bgr1, f"ext1 depth  (vmax={depth_vmax:.1f}m)",                   (5, 15), font, fs, (255,255,255), th)
    cv2.putText(depth_bgr2, f"ext2 depth  (vmax={depth_vmax:.1f}m)",                   (5, 15), font, fs, (255,255,255), th)

    top    = np.concatenate([panel1,     panel2],     axis=1)
    bottom = np.concatenate([depth_bgr1, depth_bgr2], axis=1)
    return np.concatenate([top, bottom], axis=0)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir",  type=str,
                   default="/work/hdd/bgkz/droid_raw_large_superset")
    p.add_argument("--raw-dir",      type=str, required=True)
    p.add_argument("--depth-dir",    type=str, required=True)
    p.add_argument("--canonical-id", type=str, required=True)
    p.add_argument("--cam2cam-json", type=str, default=None,
                   help="Path to cam2cam_extrinsics.json. When provided, produces "
                        "two videos: {stem}_metadata.mp4 and {stem}_cam2cam.mp4")
    p.add_argument("--output",       type=str, default="eef_projection.mp4")
    p.add_argument("--fps",          type=int, default=15)
    p.add_argument("--depth-vmax",   type=float, default=2.0,
                   help="Depth value (metres) that maps to full colormap saturation")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    raw_dir     = Path(args.raw_dir)
    depth_dir   = Path(args.depth_dir)
    cid         = args.canonical_id

    # ── Metadata extrinsics + intrinsics ─────────────────────────────────────
    meta        = load_meta(raw_dir, cid)
    ext1_serial = str(meta["ext1_cam_serial"])
    ext2_serial = str(meta["ext2_cam_serial"])

    T1_meta  = dof_to_mat(meta["ext1_cam_extrinsics"])
    T2_meta  = dof_to_mat(meta["ext2_cam_extrinsics"])
    intr1_meta = np.load(depth_dir / cid / ext1_serial / "intrinsics.npy")
    intr2_meta = np.load(depth_dir / cid / ext2_serial / "intrinsics.npy")

    print(f"ext1 serial: {ext1_serial}  intrinsics: {np.round(intr1_meta, 2)}")
    print(f"ext2 serial: {ext2_serial}  intrinsics: {np.round(intr2_meta, 2)}")

    # ── Cam2cam annotation extrinsics (optional) ──────────────────────────────
    use_cam2cam = args.cam2cam_json is not None
    if use_cam2cam:
        T1_c2c, T2_c2c, intr1_c2c, intr2_c2c = load_cam2cam(
            Path(args.cam2cam_json), cid)
        print(f"cam2cam intrinsics ext1: {np.round(intr1_c2c, 2)}")
        print(f"cam2cam intrinsics ext2: {np.round(intr2_c2c, 2)}")

    # ── EEF positions ─────────────────────────────────────────────────────────
    eef_positions = load_eef_positions(raw_dir, cid)
    print(f"EEF positions: {eef_positions.shape}  "
          f"range: {eef_positions.min(0).round(3)} → {eef_positions.max(0).round(3)}")

    # ── Find mp4s ─────────────────────────────────────────────────────────────
    id_map_path = dataset_dir / "meta" / "episode_index_to_id.json"
    if not id_map_path.exists():
        raise ValueError(f"episode_index_to_id.json not found at {id_map_path}")
    with open(id_map_path) as f:
        id_map = json.load(f)
    reverse = {(v["canonical_id"] if isinstance(v, dict) else v): int(k)
               for k, v in id_map.items()}
    if cid not in reverse:
        raise ValueError(f"canonical_id {cid!r} not in episode_index_to_id.json")
    episode_idx = reverse[cid]
    chunk = episode_idx // 1000

    mp4_ext1 = (dataset_dir / "videos" / f"chunk-{chunk:03d}"
                / "observation.images.exterior_image_1_left"
                / f"episode_{episode_idx:06d}.mp4")
    mp4_ext2 = (dataset_dir / "videos" / f"chunk-{chunk:03d}"
                / "observation.images.exterior_image_2_left"
                / f"episode_{episode_idx:06d}.mp4")

    if not mp4_ext1.exists():
        raise ValueError(f"ext1 mp4 not found: {mp4_ext1}")
    if not mp4_ext2.exists():
        raise ValueError(f"ext2 mp4 not found: {mp4_ext2}")

    # ── Decode frames + depth ─────────────────────────────────────────────────
    print("Decoding ext1 frames ...")
    frames_ext1 = decode_all_frames(mp4_ext1)
    print("Decoding ext2 frames ...")
    frames_ext2 = decode_all_frames(mp4_ext2)

    print("Loading depth arrays ...")
    depth_arr1 = load_depth(depth_dir, cid, ext1_serial)
    depth_arr2 = load_depth(depth_dir, cid, ext2_serial)
    if depth_arr1 is None:
        print(f"  WARNING: no depth for ext1 ({ext1_serial})")
    if depth_arr2 is None:
        print(f"  WARNING: no depth for ext2 ({ext2_serial})")

    n_frames = min(len(frames_ext1), len(frames_ext2), len(eef_positions))
    print(f"Rendering {n_frames} frames ...")

    # ── Output writers ────────────────────────────────────────────────────────
    out_size = (_IMG_W * 2, _IMG_H * 2)
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")

    out_stem = Path(args.output).with_suffix("")
    out_ext  = Path(args.output).suffix or ".mp4"

    if use_cam2cam:
        path_meta  = str(out_stem) + "_metadata" + out_ext
        path_cam2cam = str(out_stem) + "_cam2cam"  + out_ext
        writer_meta   = cv2.VideoWriter(path_meta,   fourcc, args.fps, out_size)
        writer_cam2cam = cv2.VideoWriter(path_cam2cam, fourcc, args.fps, out_size)
        print(f"Writing: {path_meta}")
        print(f"         {path_cam2cam}")
    else:
        path_meta  = args.output
        writer_meta = cv2.VideoWriter(path_meta, fourcc, args.fps, out_size)
        writer_cam2cam = None
        print(f"Writing: {path_meta}")

    dark_panel = np.full((_IMG_H, _IMG_W, 3), 40, dtype=np.uint8)

    for fi in range(n_frames):
        img1 = cv2.resize(frames_ext1[fi], (_IMG_W, _IMG_H))
        img2 = cv2.resize(frames_ext2[fi], (_IMG_W, _IMG_H))
        bgr1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        bgr2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        eef = eef_positions[fi]

        # Depth frames (same for both videos)
        d1 = depth_arr1[min(fi, depth_arr1.shape[0] - 1)] if depth_arr1 is not None else None
        d2 = depth_arr2[min(fi, depth_arr2.shape[0] - 1)] if depth_arr2 is not None else None
        depth_bgr1 = depth_to_bgr(d1, vmax=args.depth_vmax) if d1 is not None else dark_panel.copy()
        depth_bgr2 = depth_to_bgr(d2, vmax=args.depth_vmax) if d2 is not None else dark_panel.copy()

        # Metadata video
        frame_meta = render_frame(
            bgr1, bgr2, depth_bgr1, depth_bgr2,
            d1, d2, img1, img2,
            eef, T1_meta, T2_meta, intr1_meta, intr2_meta,
            T1_meta, T2_meta, intr1_meta, intr2_meta,
            source_label="metadata", depth_vmax=args.depth_vmax, frame_idx=fi,
        )
        writer_meta.write(frame_meta)

        # Cam2cam video — cross-projection uses cam2cam extrinsics,
        # but EEF dot always uses metadata (cam2cam poses are not in robot base frame)
        if use_cam2cam and writer_cam2cam is not None:
            frame_c2c = render_frame(
                bgr1, bgr2, depth_bgr1, depth_bgr2,
                d1, d2, img1, img2,
                eef, T1_c2c, T2_c2c, intr1_c2c, intr2_c2c,
                T1_meta, T2_meta, intr1_meta, intr2_meta,
                source_label="cam2cam", depth_vmax=args.depth_vmax, frame_idx=fi,
            )
            writer_cam2cam.write(frame_c2c)

    writer_meta.release()
    if writer_cam2cam is not None:
        writer_cam2cam.release()

    print(f"\nSaved:")
    print(f"  {path_meta}")
    if use_cam2cam:
        print(f"  {path_cam2cam}")


if __name__ == "__main__":
    main()
