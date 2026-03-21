"""
eval_depth_consistency.py

Evaluates extrinsic calibration quality via depth transfer consistency.

For each episode:
  1. Load extrinsics (metadata or cam2cam) + intrinsics
  2. For each valid depth pixel in ext1:
     - Unproject to world frame using ext1 extrinsics
     - Project into ext2 frame → predicted depth d_pred at pixel (u2, v2)
     - Look up actual depth d_actual at (u2, v2) in ext2
     - Occlusion check: skip if d_actual < d_pred - eps (something in front)
     - Accept if d_actual is within [d_pred - eps, d_pred + accept_thresh]
     - Error = |d_actual - d_pred| in cm for accepted pixels
  3. Report: mean/median depth error, % pixels accepted

If extrinsics are correct → small depth error on accepted pixels.
If extrinsics are wrong → very few accepted pixels or large errors.

Usage:
    python data_processing/eval_depth_consistency.py \
        --raw-dir /work/nvme/bgkz/droid_multilab_raw \
        --depth-dir /work/nvme/bgkz/droid_multilab_depths \
        --extr-source metadata

    python data_processing/eval_depth_consistency.py \
        --raw-dir /work/nvme/bgkz/droid_multilab_raw \
        --depth-dir /work/nvme/bgkz/droid_multilab_depths \
        --cam2cam-json /work/nvme/bgkz/droid_annotations/cam2cam_extrinsics.json \
        --extr-source cam2cam
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import blosc
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

_IMG_H = 180
_IMG_W = 320
_SCALE_X = _IMG_W / 1280
_SCALE_Y  = _IMG_H / 720


# ── Extrinsics / intrinsics ───────────────────────────────────────────────────

def dof_to_mat(dof) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler("xyz", dof[3:]).as_matrix()
    T[:3, 3]  = dof[:3]
    return T


def load_from_metadata(raw_dir: Path, cid: str):
    meta_files = list((raw_dir / cid).glob("metadata_*.json"))
    if not meta_files:
        raise ValueError(f"No metadata_*.json in {raw_dir/cid}")
    meta = json.loads(meta_files[0].read_text())
    s1 = str(meta["ext1_cam_serial"])
    s2 = str(meta["ext2_cam_serial"])
    return dof_to_mat(meta["ext1_cam_extrinsics"]), dof_to_mat(meta["ext2_cam_extrinsics"]), s1, s2


def load_from_cam2cam(cam2cam: dict, cid: str, raw_dir: Path):
    if cid not in cam2cam:
        raise ValueError(f"{cid} not in cam2cam_extrinsics.json")
    ep = cam2cam[cid]
    if "left_cam" not in ep or "right_cam" not in ep:
        raise ValueError(f"Missing left_cam/right_cam for {cid}")
    T1 = np.array(ep["left_cam"]["pose"],  dtype=np.float64)
    T2 = np.array(ep["right_cam"]["pose"], dtype=np.float64)

    def cam_K(c):
        f = c["focal"]; cx, cy = c["principal_point"]
        return np.array([[f * _SCALE_X, 0, cx * _SCALE_X],
                         [0, f * _SCALE_Y, cy * _SCALE_Y],
                         [0, 0, 1]], dtype=np.float64)

    meta_files = list((raw_dir / cid).glob("metadata_*.json"))
    if not meta_files:
        raise ValueError(f"No metadata_*.json in {raw_dir/cid}")
    meta = json.loads(meta_files[0].read_text())
    s1 = str(meta["ext1_cam_serial"])
    s2 = str(meta["ext2_cam_serial"])
    return T1, T2, s1, s2, cam_K(ep["left_cam"]), cam_K(ep["right_cam"])


def load_K(depth_dir: Path, cid: str, serial: str) -> np.ndarray:
    intr = np.load(depth_dir / cid / serial / "intrinsics.npy")
    fx, fy, cx, cy = intr
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def load_depth_frame(depth_dir: Path, cid: str, serial: str, frame_idx: int) -> np.ndarray:
    ep_dir = depth_dir / cid / serial
    shape = np.load(ep_dir / "shape.npy")
    raw   = (ep_dir / "depth.blosc").read_bytes()
    arr   = np.frombuffer(blosc.decompress(raw), dtype=np.float32).reshape(shape)
    return arr[min(frame_idx, arr.shape[0] - 1)]


# ── Depth transfer ────────────────────────────────────────────────────────────

def depth_transfer_error(
    depth1: np.ndarray,   # (H, W) depth in ext1 at working res
    depth2: np.ndarray,   # (H, W) depth in ext2 at working res
    T1: np.ndarray,       # 4×4 ext1 cam→world
    T2: np.ndarray,       # 4×4 ext2 cam→world
    K1: np.ndarray,       # 3×3 intrinsics ext1
    K2: np.ndarray,       # 3×3 intrinsics ext2
    max_depth: float = 2.0,
    eps: float = 0.05,        # occlusion tolerance (m)
    accept_thresh: float = 0.5,  # max depth diff to count as same surface (m)
    max_pts: int = 5000,
) -> dict | None:
    """
    Transfer depth pixels from ext1 → ext2 and compare with ext2's depth.

    Returns dict with mean/median error (cm), coverage (fraction of ext1 depth
    pixels that found a consistent match in ext2), or None if no valid pixels.
    """
    d1 = cv2.resize(depth1, (_IMG_W, _IMG_H), interpolation=cv2.INTER_NEAREST)
    d2 = cv2.resize(depth2, (_IMG_W, _IMG_H), interpolation=cv2.INTER_NEAREST)

    # Valid ext1 depth pixels
    valid1 = np.isfinite(d1) & (d1 > 0.05) & (d1 < max_depth)
    ys, xs = np.where(valid1)
    if len(ys) == 0:
        return None

    # Subsample for speed
    if len(ys) > max_pts:
        idx = np.random.choice(len(ys), max_pts, replace=False)
        ys, xs = ys[idx], xs[idx]

    D = d1[ys, xs]
    fx1, fy1, cx1, cy1 = K1[0,0], K1[1,1], K1[0,2], K1[1,2]

    # Unproject ext1 pixels → world
    pts_cam1 = np.stack([(xs - cx1) * D / fx1,
                          (ys - cy1) * D / fy1,
                          D,
                          np.ones_like(D)], axis=-1)           # (N, 4)
    pts_world = (pts_cam1 @ T1.T)[:, :3]                       # (N, 3)

    # Project world points → ext2 camera frame
    T2_inv = np.linalg.inv(T2)
    pts_hom = np.hstack([pts_world, np.ones((len(pts_world), 1))])
    pts_cam2 = (pts_hom @ T2_inv.T)[:, :3]                     # (N, 3)

    # Discard points behind ext2
    in_front = pts_cam2[:, 2] > 0
    pts_cam2 = pts_cam2[in_front]
    D_pred   = pts_cam2[:, 2]

    fx2, fy2, cx2, cy2 = K2[0,0], K2[1,1], K2[0,2], K2[1,2]
    u2 = (fx2 * pts_cam2[:, 0] / pts_cam2[:, 2] + cx2).astype(int)
    v2 = (fy2 * pts_cam2[:, 1] / pts_cam2[:, 2] + cy2).astype(int)

    # Keep only pixels that fall within ext2's image
    in_img = (u2 >= 0) & (u2 < _IMG_W) & (v2 >= 0) & (v2 < _IMG_H)
    u2, v2, D_pred = u2[in_img], v2[in_img], D_pred[in_img]

    if len(u2) == 0:
        return None

    # Look up actual ext2 depth at projected pixels
    D_actual = d2[v2, u2].astype(np.float64)

    # Filter: need valid ext2 depth
    valid2 = np.isfinite(D_actual) & (D_actual > 0.05) & (D_actual < max_depth)
    D_actual = D_actual[valid2]
    D_pred   = D_pred[valid2]

    if len(D_actual) == 0:
        return None

    diff = D_actual - D_pred   # positive = further away in ext2 than predicted

    # Occlusion check: skip pixels where ext2 sees something closer (occluded)
    not_occluded = diff > -eps

    # Accept pixels where the depth difference is within accept_thresh
    accepted = not_occluded & (diff < accept_thresh)

    n_total    = len(ys)          # total ext1 depth pixels sampled
    n_in_ext2  = len(u2)         # projected into ext2 image
    n_valid2   = int(valid2.sum())
    n_accepted = int(accepted.sum())

    if n_accepted == 0:
        return {
            "mean_cm":    None,
            "median_cm":  None,
            "coverage":   0.0,
            "n_accepted": 0,
            "n_total":    n_total,
        }

    errors_cm = np.abs(diff[accepted]) * 100  # m → cm

    return {
        "mean_cm":    float(np.mean(errors_cm)),
        "median_cm":  float(np.median(errors_cm)),
        "coverage":   n_accepted / n_total,
        "n_accepted": n_accepted,
        "n_total":    n_total,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir",       type=str, required=True)
    p.add_argument("--depth-dir",     type=str, required=True)
    p.add_argument("--extr-source",   type=str, default="metadata",
                   choices=["metadata", "cam2cam"])
    p.add_argument("--cam2cam-json",  type=str, default=None)
    p.add_argument("--frame",         type=int, default=30,
                   help="Which frame to evaluate (default 30)")
    p.add_argument("--max-depth",     type=float, default=2.0)
    p.add_argument("--eps",           type=float, default=0.05,
                   help="Occlusion tolerance in metres (default 0.05)")
    p.add_argument("--accept-thresh", type=float, default=0.5,
                   help="Max depth diff (m) to count as same surface (default 0.5)")
    return p.parse_args()


def lab_of(cid): return cid.split("+")[0]


def print_lab_table(lab_results: dict):
    print(f"\n{'Lab':<12}  {'N':>5}  {'mean cm':>8}  {'median cm':>9}  {'coverage':>9}  {'no_data':>8}")
    print("-" * 60)
    all_means, all_meds, all_covs = [], [], []
    for lab in sorted(lab_results):
        recs = [r for r in lab_results[lab] if r is not None]
        no_data = len(lab_results[lab]) - len(recs)
        valid = [r for r in recs if r["mean_cm"] is not None]
        if valid:
            means = [r["mean_cm"]   for r in valid]
            meds  = [r["median_cm"] for r in valid]
            covs  = [r["coverage"]  for r in valid]
            all_means.extend(means); all_meds.extend(meds); all_covs.extend(covs)
            print(f"{lab:<12}  {len(lab_results[lab]):>5}  {np.mean(means):8.1f}  "
                  f"{np.mean(meds):9.1f}  {np.mean(covs):9.1%}  {no_data:>8}")
        else:
            print(f"{lab:<12}  {len(lab_results[lab]):>5}  {'—':>8}  {'—':>9}  {'—':>9}  {no_data:>8}")
    print("-" * 60)
    if all_means:
        print(f"{'ALL':<12}  {'':>5}  {np.mean(all_means):8.1f}  "
              f"{np.mean(all_meds):9.1f}  {np.mean(all_covs):9.1%}")


def main():
    args = parse_args()
    raw_dir   = Path(args.raw_dir)
    depth_dir = Path(args.depth_dir)

    cam2cam = None
    if args.extr_source == "cam2cam":
        if not args.cam2cam_json:
            raise ValueError("--cam2cam-json required when --extr-source=cam2cam")
        print("Loading cam2cam_extrinsics.json ...")
        with open(args.cam2cam_json) as f:
            cam2cam = json.load(f)

    episodes = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())
    print(f"Evaluating {len(episodes)} episodes  (source={args.extr_source})\n")
    print(f"{'Episode':<45}  {'mean cm':>8}  {'med cm':>7}  {'coverage':>9}  {'accepted':>9}")
    print("-" * 85)

    lab_results = defaultdict(list)

    for i, cid in enumerate(episodes):
        try:
            if args.extr_source == "metadata":
                T1, T2, s1, s2 = load_from_metadata(raw_dir, cid)
                K1 = load_K(depth_dir, cid, s1)
                K2 = load_K(depth_dir, cid, s2)
            else:
                T1, T2, s1, s2, K1, K2 = load_from_cam2cam(cam2cam, cid, raw_dir)

            d1 = load_depth_frame(depth_dir, cid, s1, args.frame)
            d2 = load_depth_frame(depth_dir, cid, s2, args.frame)

            r = depth_transfer_error(d1, d2, T1, T2, K1, K2,
                                     max_depth=args.max_depth,
                                     eps=args.eps,
                                     accept_thresh=args.accept_thresh)
            lab_results[lab_of(cid)].append(r)

            if r is None:
                print(f"{cid:<45}  (no valid depth)")
            elif r["mean_cm"] is None:
                print(f"{cid:<45}  (no accepted pixels)  coverage=0")
            else:
                print(f"{cid:<45}  {r['mean_cm']:8.1f}  {r['median_cm']:7.1f}  "
                      f"{r['coverage']:9.1%}  {r['n_accepted']:>4}/{r['n_total']:>4}")

        except Exception as e:
            lab_results[lab_of(cid)].append(None)
            print(f"{cid:<45}  ERROR: {e}")

        if (i + 1) % 200 == 0:
            print(f"\n[{i+1}/{len(episodes)} done]")
            print_lab_table(lab_results)
            print()

    print_lab_table(lab_results)


if __name__ == "__main__":
    main()
