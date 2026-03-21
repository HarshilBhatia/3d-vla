"""
eval_extrinsics.py

Measures extrinsic calibration quality across all episodes in droid_multilab_raw
via epipolar geometry. Reports per-episode and per-lab statistics.

For each episode:
  1. Load extrinsics + intrinsics
  2. SIFT-match ext1 vs ext2 at frame 30
  3. Compute mean/median symmetric epipolar error on all ratio-filtered matches
     (no RANSAC dependency — episode is never skipped due to RANSAC failure)

Extrinsics source (--extr-source):
  metadata : per-episode metadata_*.json  [unreliable]
  cam2cam  : cam2cam_extrinsics.json      [calibrated, left=ext1 right=ext2]

Usage:
    python data_processing/eval_extrinsics.py \
        --dataset-dir /work/nvme/bgkz/droid_raw_large_superset \
        --raw-dir /work/nvme/bgkz/droid_multilab_raw \
        --depth-dir /work/nvme/bgkz/droid_multilab_depths \
        --extr-source metadata

    python data_processing/eval_extrinsics.py \
        --dataset-dir /work/nvme/bgkz/droid_raw_large_superset \
        --raw-dir /work/nvme/bgkz/droid_multilab_raw \
        --depth-dir /work/nvme/bgkz/droid_multilab_depths \
        --cam2cam-json /work/nvme/bgkz/droid_annotations/cam2cam_extrinsics.json \
        --extr-source cam2cam
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import av
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

_IMG_H = 180
_IMG_W = 320
_SCALE_X = _IMG_W / 1280
_SCALE_Y = _IMG_H / 720


# ── Extrinsics loading ────────────────────────────────────────────────────────

def dof_to_mat(dof) -> np.ndarray:
    """[tx, ty, tz, rx, ry, rz] → 4×4 cam→base (Euler XYZ)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler("xyz", dof[3:]).as_matrix()
    T[:3, 3] = dof[:3]
    return T


def load_from_metadata(raw_dir: Path, cid: str):
    """Returns T1, T2, K1, K2 from metadata_*.json + intrinsics.npy."""
    ep_dir = raw_dir / cid
    meta_files = list(ep_dir.glob("metadata_*.json"))
    if not meta_files:
        raise ValueError(f"No metadata_*.json in {ep_dir}")
    meta = json.loads(meta_files[0].read_text())
    s1 = str(meta["ext1_cam_serial"])
    s2 = str(meta["ext2_cam_serial"])
    T1 = dof_to_mat(meta["ext1_cam_extrinsics"])
    T2 = dof_to_mat(meta["ext2_cam_extrinsics"])
    return T1, T2, s1, s2


def load_from_cam2cam(cam2cam: dict, cid: str, raw_dir: Path):
    """
    Returns T1, T2, K1, K2 from cam2cam_extrinsics.json.
    left_cam = ext1, right_cam = ext2 (confirmed by focal length matching).
    Intrinsics at native res, scaled to _IMG_H × _IMG_W.
    """
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


# ── Fundamental matrix ────────────────────────────────────────────────────────

def skew(v) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def compute_F(T1, T2, K1, K2) -> np.ndarray:
    T12 = np.linalg.inv(T2) @ T1
    E = skew(T12[:3, 3]) @ T12[:3, :3]
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)


def sym_epipolar_error(F, p1h, p2h) -> float:
    l2 = F @ p1h;  l1 = F.T @ p2h
    d2 = abs(p2h @ l2) / (np.sqrt(l2[0]**2 + l2[1]**2) + 1e-9)
    d1 = abs(p1h @ l1) / (np.sqrt(l1[0]**2 + l1[1]**2) + 1e-9)
    return float((d1 + d2) / 2)


# ── Frame decode + SIFT ───────────────────────────────────────────────────────

def decode_nth_frame(mp4_path: Path, n: int = 30) -> np.ndarray:
    with av.open(str(mp4_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        i = 0
        for packet in container.demux(stream):
            for frame in packet.decode():
                if i >= n:
                    return frame.to_ndarray(format="rgb24")
                i += 1
    # episode shorter than n frames — return last decoded frame
    raise ValueError(f"Episode has fewer than {n} frames: {mp4_path}")


def sift_matches(gray1, gray2, ratio=0.75):
    sift = cv2.SIFT_create(nfeatures=500)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return np.empty((0, 2)), np.empty((0, 2))
    raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]
    if not good:
        return np.empty((0, 2)), np.empty((0, 2))
    pts1 = np.array([kp1[m.queryIdx].pt for m in good], dtype=np.float32)
    pts2 = np.array([kp2[m.trainIdx].pt for m in good], dtype=np.float32)
    return pts1, pts2


# ── Per-episode evaluation ────────────────────────────────────────────────────

def eval_episode(mp4_ext1, mp4_ext2, F, frame_n=30) -> dict | None:
    try:
        img1 = decode_nth_frame(mp4_ext1, n=frame_n)
        img2 = decode_nth_frame(mp4_ext2, n=frame_n)
    except ValueError:
        # Episode too short — try frame 0
        img1 = decode_nth_frame(mp4_ext1, n=0)
        img2 = decode_nth_frame(mp4_ext2, n=0)

    g1 = cv2.cvtColor(cv2.resize(img1, (_IMG_W, _IMG_H)), cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(cv2.resize(img2, (_IMG_W, _IMG_H)), cv2.COLOR_RGB2GRAY)
    pts1, pts2 = sift_matches(g1, g2)

    if len(pts1) == 0:
        return None

    pts1h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2h = np.hstack([pts2, np.ones((len(pts2), 1))])
    errs = [sym_epipolar_error(F, a, b) for a, b in zip(pts1h, pts2h)]

    return {
        "mean":      float(np.mean(errs)),
        "median":    float(np.median(errs)),
        "p90":       float(np.percentile(errs, 90)),
        "n_matches": int(len(pts1)),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir",  type=str, default="/work/nvme/bgkz/droid_raw_large_superset")
    p.add_argument("--raw-dir",      type=str, required=True)
    p.add_argument("--depth-dir",    type=str, required=True)
    p.add_argument("--extr-source",  type=str, default="metadata",
                   choices=["metadata", "cam2cam"])
    p.add_argument("--cam2cam-json", type=str, default=None)
    return p.parse_args()


def lab_of(cid: str) -> str:
    return cid.split("+")[0]


def print_lab_table(lab_results: dict):
    print(f"\n{'Lab':<15}  {'N':>5}  {'mean':>8}  {'median':>8}  {'p90':>8}  {'no_match':>9}")
    print("-" * 62)
    all_means, all_meds, all_p90s = [], [], []
    for lab in sorted(lab_results):
        recs = lab_results[lab]
        valid = [r for r in recs if r is not None]
        no_match = len(recs) - len(valid)
        if valid:
            means  = [r["mean"]   for r in valid]
            meds   = [r["median"] for r in valid]
            p90s   = [r["p90"]    for r in valid]
            all_means.extend(means); all_meds.extend(meds); all_p90s.extend(p90s)
            print(f"{lab:<15}  {len(recs):>5}  {np.mean(means):8.2f}  "
                  f"{np.mean(meds):8.2f}  {np.mean(p90s):8.2f}  {no_match:>9}")
        else:
            print(f"{lab:<15}  {len(recs):>5}  {'—':>8}  {'—':>8}  {'—':>8}  {no_match:>9}")
    print("-" * 62)
    if all_means:
        print(f"{'ALL':<15}  {'':>5}  {np.mean(all_means):8.2f}  "
              f"{np.mean(all_meds):8.2f}  {np.mean(all_p90s):8.2f}")


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    raw_dir     = Path(args.raw_dir)
    depth_dir   = Path(args.depth_dir)

    cam2cam = None
    if args.extr_source == "cam2cam":
        if not args.cam2cam_json:
            raise ValueError("--cam2cam-json required when --extr-source=cam2cam")
        print("Loading cam2cam_extrinsics.json ...")
        with open(args.cam2cam_json) as f:
            cam2cam = json.load(f)

    # Build episode list from droid_multilab_raw (all 1400 episodes)
    with open(dataset_dir / "meta" / "episode_index_to_id.json") as f:
        id_map = json.load(f)
    reverse = {(v["canonical_id"] if isinstance(v, dict) else v): int(k)
               for k, v in id_map.items()}

    episodes = []
    for cid in sorted((raw_dir).iterdir().__iter__(), key=lambda p: p.name):
        cid = cid.name
        if cid not in reverse:
            continue
        ep_idx = reverse[cid]
        chunk  = ep_idx // 1000
        mp4_1  = (dataset_dir / "videos" / f"chunk-{chunk:03d}"
                  / "observation.images.exterior_image_1_left"
                  / f"episode_{ep_idx:06d}.mp4")
        mp4_2  = (dataset_dir / "videos" / f"chunk-{chunk:03d}"
                  / "observation.images.exterior_image_2_left"
                  / f"episode_{ep_idx:06d}.mp4")
        if mp4_1.exists() and mp4_2.exists():
            episodes.append((cid, mp4_1, mp4_2))

    print(f"Evaluating {len(episodes)} episodes  (source={args.extr_source})\n")
    print(f"{'Episode':<45}  {'mean':>8}  {'median':>8}  {'p90':>8}  {'matches':>8}")
    print("-" * 85)

    lab_results = defaultdict(list)

    for i, (cid, mp4_1, mp4_2) in enumerate(episodes):
        try:
            if args.extr_source == "metadata":
                T1, T2, s1, s2 = load_from_metadata(raw_dir, cid)
                K1 = load_K(depth_dir, cid, s1)
                K2 = load_K(depth_dir, cid, s2)
            else:
                T1, T2, s1, s2, K1, K2 = load_from_cam2cam(cam2cam, cid, raw_dir)

            F = compute_F(T1, T2, K1, K2)
            r = eval_episode(mp4_1, mp4_2, F)

            lab_results[lab_of(cid)].append(r)

            if r is None:
                print(f"{cid:<45}  (no matches)")
            else:
                print(f"{cid:<45}  {r['mean']:8.2f}  {r['median']:8.2f}  "
                      f"{r['p90']:8.2f}  {r['n_matches']:>8}")

        except Exception as e:
            lab_results[lab_of(cid)].append(None)
            print(f"{cid:<45}  ERROR: {e}")

        if (i + 1) % 100 == 0:
            print(f"\n[{i+1}/{len(episodes)} done — intermediate lab summary]")
            print_lab_table(lab_results)
            print()

    print_lab_table(lab_results)


if __name__ == "__main__":
    main()
