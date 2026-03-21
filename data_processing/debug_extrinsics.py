"""
debug_extrinsics.py

Tries all 4 combos of T vs inv(T) for ext1 + ext2 and ranks them by
Z-distribution alignment across multiple frames.

Key insight: both ext cameras see the same robot workspace, so in the correct
configuration they should observe the same "floor/table" Z level. We compare
their Z-distributions (p10, median, p90) — lower difference = better.

Usage:
    python data_processing/debug_extrinsics.py \
        --raw-dir /work/nvme/bgkz/droid_multilab_raw \
        --depth-dir /work/nvme/bgkz/droid_multilab_depths \
        --canonical-id AUTOLab+0d4edc83+2023-10-21-19h-08m-41s
"""

import argparse
import json
from pathlib import Path

import blosc
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

_DEPTH_H = 180
_DEPTH_W = 320


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_depth_frame(depth_dir, canonical_id, serial, frame_idx):
    ep_dir = depth_dir / canonical_id / serial
    shape  = np.load(ep_dir / "shape.npy")
    raw    = (ep_dir / "depth.blosc").read_bytes()
    arr    = np.frombuffer(blosc.decompress(raw), dtype=np.float32).reshape(shape)
    return arr[min(frame_idx, arr.shape[0] - 1)]


def load_intrinsics(depth_dir, canonical_id, serial):
    return np.load(depth_dir / canonical_id / serial / "intrinsics.npy")


def load_meta(raw_dir, canonical_id):
    ep_dir = raw_dir / canonical_id
    meta_files = list(ep_dir.glob("metadata_*.json"))
    if not meta_files:
        raise ValueError(f"No metadata_*.json in {ep_dir}")
    return json.loads(meta_files[0].read_text())


def dof_to_mat(dof, rot_convention="euler_xyz"):
    T = np.eye(4, dtype=np.float64)
    if rot_convention == "euler_xyz":
        T[:3, :3] = Rotation.from_euler("xyz", dof[3:]).as_matrix()
    elif rot_convention == "euler_XYZ":
        T[:3, :3] = Rotation.from_euler("XYZ", dof[3:]).as_matrix()
    elif rot_convention == "euler_zyx":
        T[:3, :3] = Rotation.from_euler("zyx", dof[3:]).as_matrix()
    elif rot_convention == "euler_ZYX":
        T[:3, :3] = Rotation.from_euler("ZYX", dof[3:]).as_matrix()
    elif rot_convention == "rotvec":
        T[:3, :3] = Rotation.from_rotvec(dof[3:]).as_matrix()
    else:
        raise ValueError(f"Unknown rot_convention: {rot_convention}")
    T[:3, 3] = dof[:3]
    return T


# ── Unprojection (camera-frame depth < 3m to avoid background noise) ──────────

def unproject(depth_frame, intrinsics, T_cam2world, max_pts=8000):
    depth = cv2.resize(depth_frame, (_DEPTH_W, _DEPTH_H), interpolation=cv2.INTER_NEAREST)
    fx, fy, cx, cy = intrinsics
    valid = np.isfinite(depth) & (depth > 0.1) & (depth < 3.0)  # filter in camera frame
    ys, xs = np.where(valid)
    D = depth[ys, xs]
    pts_cam = np.stack([(xs - cx) * D / fx, (ys - cy) * D / fy, D, np.ones_like(D)], axis=-1)
    pts_world = (pts_cam @ T_cam2world.T)[:, :3].astype(np.float32)
    if len(pts_world) > max_pts:
        pts_world = pts_world[np.random.choice(len(pts_world), max_pts, replace=False)]
    return pts_world


# ── Per-frame score: Z-distribution difference + XY plausibility ──────────────

def frame_score(pts1, pts2):
    if len(pts1) < 50 or len(pts2) < 50:
        return None  # skip frames with too few valid points

    z1 = np.percentile(pts1[:, 2], [10, 50, 90])
    z2 = np.percentile(pts2[:, 2], [10, 50, 90])
    z_diff = float(np.abs(z1 - z2).mean())

    # XY centroids should be in the same rough region (< a few meters apart)
    xy_dist = float(np.linalg.norm(pts1[:, :2].mean(0) - pts2[:, :2].mean(0)))

    # Plausibility: median distance from origin should be < 2m
    med1 = float(np.median(np.linalg.norm(pts1, axis=1)))
    med2 = float(np.median(np.linalg.norm(pts2, axis=1)))
    implausibility = max(0.0, med1 - 2.0) + max(0.0, med2 - 2.0)

    return z_diff + xy_dist * 0.3 + implausibility * 3.0


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir",      type=str, required=True)
    p.add_argument("--depth-dir",    type=str, required=True)
    p.add_argument("--canonical-id", type=str, required=True)
    p.add_argument("--n-frames",     type=int, default=10,
                   help="Number of frames to average score over (default: 10)")
    return p.parse_args()


def main():
    args      = parse_args()
    raw_dir   = Path(args.raw_dir)
    depth_dir = Path(args.depth_dir)
    cid       = args.canonical_id

    meta = load_meta(raw_dir, cid)
    ext1_serial = str(meta["ext1_cam_serial"])
    ext2_serial = str(meta["ext2_cam_serial"])
    dof1 = np.array(meta["ext1_cam_extrinsics"], dtype=np.float64)
    dof2 = np.array(meta["ext2_cam_extrinsics"], dtype=np.float64)

    print(f"ext1 serial: {ext1_serial}  DOF: {np.round(dof1, 3)}")
    print(f"ext2 serial: {ext2_serial}  DOF: {np.round(dof2, 3)}")

    intr1 = load_intrinsics(depth_dir, cid, ext1_serial)
    intr2 = load_intrinsics(depth_dir, cid, ext2_serial)

    # Load all frames needed
    depth_ep1 = depth_dir / cid / ext1_serial / "shape.npy"
    n_frames  = int(np.load(depth_ep1)[0])
    frame_indices = np.linspace(0, n_frames - 1, args.n_frames, dtype=int)
    print(f"Evaluating {len(frame_indices)} frames: {frame_indices}\n")

    rot_conventions = ["euler_xyz", "euler_XYZ", "euler_zyx", "euler_ZYX", "rotvec"]

    configs = []
    for rot_conv in rot_conventions:
        T1_base = dof_to_mat(dof1, rot_conv)
        T2_base = dof_to_mat(dof2, rot_conv)
        for inv1, inv2 in [(False, False), (True, False), (False, True), (True, True)]:
            T1 = np.linalg.inv(T1_base) if inv1 else T1_base
            T2 = np.linalg.inv(T2_base) if inv2 else T2_base
            label = f"rot={rot_conv:<12} inv1={inv1} inv2={inv2}"
            configs.append((label, T1, T2))

    results = []
    for label, T1, T2 in configs:
        scores = []
        for fi in frame_indices:
            d1 = load_depth_frame(depth_dir, cid, ext1_serial, int(fi))
            d2 = load_depth_frame(depth_dir, cid, ext2_serial, int(fi))
            pts1 = unproject(d1, intr1, T1)
            pts2 = unproject(d2, intr2, T2)
            s = frame_score(pts1, pts2)
            if s is not None:
                scores.append(s)

        if not scores:
            results.append((1e9, label, {}))
            continue

        mean_score = float(np.mean(scores))
        std_score  = float(np.std(scores))

        # Compute summary stats for best config printout
        d1 = load_depth_frame(depth_dir, cid, ext1_serial, int(frame_indices[len(frame_indices)//2]))
        d2 = load_depth_frame(depth_dir, cid, ext2_serial, int(frame_indices[len(frame_indices)//2]))
        pts1 = unproject(d1, intr1, T1)
        pts2 = unproject(d2, intr2, T2)
        z1 = np.percentile(pts1[:,2], [10, 50, 90])
        z2 = np.percentile(pts2[:,2], [10, 50, 90])

        results.append((mean_score, label, {
            "mean_score": mean_score,
            "std_score":  std_score,
            "n_frames":   len(scores),
            "z_p10_50_90_ext1": z1.tolist(),
            "z_p10_50_90_ext2": z2.tolist(),
            "centroid1":  pts1.mean(0).tolist(),
            "centroid2":  pts2.mean(0).tolist(),
        }))

    results.sort(key=lambda x: x[0])

    print(f"{'Rank':<5} {'Score':>8}  {'±Std':>6}  {'Frames':>6}  Config")
    print("-" * 60)
    for rank, (score, label, m) in enumerate(results, 1):
        std = m.get("std_score", 0)
        n   = m.get("n_frames", 0)
        print(f"{rank:<5} {score:>8.3f}  {std:>6.3f}  {n:>6}  {label}")

    print()
    _, best_label, best_m = results[0]
    print(f"Best config: {best_label}")
    if best_m:
        print(f"  ext1 Z [p10, med, p90]: {np.round(best_m['z_p10_50_90_ext1'], 3)}")
        print(f"  ext2 Z [p10, med, p90]: {np.round(best_m['z_p10_50_90_ext2'], 3)}")
        print(f"  ext1 centroid XYZ: {np.round(best_m['centroid1'], 3)}")
        print(f"  ext2 centroid XYZ: {np.round(best_m['centroid2'], 3)}")


if __name__ == "__main__":
    main()
