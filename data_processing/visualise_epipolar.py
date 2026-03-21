"""
visualise_epipolar.py

Validates camera extrinsics via epipolar geometry.

Computes the fundamental matrix F from our stored extrinsics, then:
  - Detects SIFT keypoints in ext1 and ext2 and matches them
  - For each match: draws the source point in ext1 and the corresponding
    epipolar line in ext2 (colour-coded by epipolar error)
  - Reports mean symmetric epipolar error per frame

If extrinsics are correct, matched points should lie on (or very near)
their epipolar lines. Mean error < 1-2 px = good alignment.

Each video is 3-panel:  ext1 (source pts) | ext2 [source] F epilines | ext2 RANSAC F

Output modes:
  No --cam2cam-json  →  one video at --output (metadata extrinsics)
  With --cam2cam-json → two videos:
      {stem}_metadata.mp4  (metadata extrinsics)
      {stem}_cam2cam.mp4   (cam2cam annotation extrinsics)

Usage:
    # metadata only
    python data_processing/visualise_epipolar.py \
        --dataset-dir /work/nvme/bgkz/droid_raw_large_superset \
        --raw-dir /work/nvme/bgkz/droid_multilab_raw \
        --depth-dir /work/nvme/bgkz/droid_multilab_depths \
        --canonical-id ILIAD+50aee79f+2023-07-12-21h-13m-44s \
        --output epipolar.mp4

    # metadata + cam2cam annotation comparison (two separate videos)
    python data_processing/visualise_epipolar.py \
        --dataset-dir /work/nvme/bgkz/droid_raw_large_superset \
        --raw-dir /work/nvme/bgkz/droid_multilab_raw \
        --depth-dir /work/nvme/bgkz/droid_multilab_depths \
        --canonical-id ILIAD+50aee79f+2023-07-12-21h-13m-44s \
        --cam2cam-json /work/nvme/bgkz/droid_annotations/cam2cam_extrinsics.json \
        --output epipolar.mp4
"""

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

_IMG_H = 180
_IMG_W = 320
# Native resolution of DROID colour cameras (used to scale cam2cam intrinsics)
_NATIVE_W = 1280
_NATIVE_H = 720
_SCALE_X = _IMG_W / _NATIVE_W   # 0.25
_SCALE_Y = _IMG_H / _NATIVE_H   # 0.25


# ── Camera / extrinsics helpers ───────────────────────────────────────────────

def load_meta(raw_dir: Path, canonical_id: str) -> dict:
    ep_dir = raw_dir / canonical_id
    meta_files = list(ep_dir.glob("metadata_*.json"))
    if not meta_files:
        raise ValueError(f"No metadata_*.json in {ep_dir}")
    return json.loads(meta_files[0].read_text())


def dof_to_mat(dof) -> np.ndarray:
    """[tx, ty, tz, rx, ry, rz] → 4×4 cam→base (Euler XYZ)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler("xyz", dof[3:]).as_matrix()
    T[:3, 3] = dof[:3]
    return T


def intrinsics_matrix(intr) -> np.ndarray:
    """[fx, fy, cx, cy] → 3×3 K."""
    fx, fy, cx, cy = intr
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def load_cam2cam(cam2cam_json: Path, cid: str, raw_dir: Path):
    """
    Load extrinsics + intrinsics from cam2cam_extrinsics.json annotations.

    Returns (T1, T2, K1, K2) where:
      T1/T2 are 4×4 cam→base pose matrices (left_cam=ext1, right_cam=ext2)
      K1/K2 are 3×3 intrinsics scaled to _IMG_H × _IMG_W

    Raises ValueError if the episode is missing from the json.
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
        raise ValueError(f"Expected 4×4 pose matrices for {cid}, got {T1.shape} / {T2.shape}")

    def _K(c):
        focal = c["focal"]
        cx, cy = c["principal_point"]
        return np.array([
            [focal * _SCALE_X,            0, cx * _SCALE_X],
            [           0,  focal * _SCALE_Y, cy * _SCALE_Y],
            [           0,            0,              1     ],
        ], dtype=np.float64)

    return T1, T2, _K(ep["left_cam"]), _K(ep["right_cam"])


def skew(v) -> np.ndarray:
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]], dtype=np.float64)


def compute_fundamental(T_cam1_to_base, T_cam2_to_base, K1, K2) -> np.ndarray:
    """
    Compute fundamental matrix F such that for corresponding points p1, p2:
        p2.T @ F @ p1 = 0

    T_cam1_to_base, T_cam2_to_base: 4×4 cam→base transforms.
    K1, K2: 3×3 intrinsic matrices for cam1 and cam2.
    """
    # Relative pose: cam1 expressed in cam2's frame
    T_base_to_cam2 = np.linalg.inv(T_cam2_to_base)
    T_1to2 = T_base_to_cam2 @ T_cam1_to_base  # 4×4

    R = T_1to2[:3, :3]
    t = T_1to2[:3, 3]

    E = skew(t) @ R                        # Essential matrix
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F


def epiline(F, pt_hom):
    """Epipolar line in image 2 for homogeneous point pt_hom in image 1."""
    return F @ pt_hom  # [a, b, c]: ax + by + c = 0


def epipolar_error(F, pt1_hom, pt2_hom) -> float:
    """Symmetric epipolar error in pixels."""
    l2 = F @ pt1_hom
    l1 = F.T @ pt2_hom
    d2 = abs(pt2_hom @ l2) / (np.sqrt(l2[0]**2 + l2[1]**2) + 1e-9)
    d1 = abs(pt1_hom @ l1) / (np.sqrt(l1[0]**2 + l1[1]**2) + 1e-9)
    return float((d1 + d2) / 2)


def draw_epiline(img_bgr, line, color):
    """Draw epipolar line [a, b, c] across the full image width."""
    a, b, c = line
    h, w = img_bgr.shape[:2]
    if abs(b) < 1e-9:
        return
    x0, x1 = 0, w - 1
    y0 = int(round((-c - a * x0) / b))
    y1 = int(round((-c - a * x1) / b))
    cv2.line(img_bgr, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)


# ── Frame helpers ─────────────────────────────────────────────────────────────

def decode_all_frames(mp4_path: Path) -> list[np.ndarray]:
    frames = []
    with av.open(str(mp4_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for packet in container.demux(stream):
            for frame in packet.decode():
                frames.append(frame.to_ndarray(format="rgb24"))
    return frames


# ── SIFT matching ─────────────────────────────────────────────────────────────

def match_sift(gray1, gray2, max_matches=30, ratio=0.75):
    """
    Detect SIFT keypoints and return matched pixel coordinates.
    Returns pts1, pts2 as (N, 2) float32 arrays, or empty arrays if no matches.
    """
    sift = cv2.SIFT_create(nfeatures=500)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return np.empty((0, 2)), np.empty((0, 2))

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in raw:
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) > max_matches:
        good = sorted(good, key=lambda x: x.distance)[:max_matches]

    if not good:
        return np.empty((0, 2)), np.empty((0, 2))

    pts1 = np.array([kp1[m.queryIdx].pt for m in good], dtype=np.float32)
    pts2 = np.array([kp2[m.trainIdx].pt for m in good], dtype=np.float32)
    return pts1, pts2


def ransac_fundamental(pts1, pts2):
    """
    Estimate fundamental matrix from matched points using RANSAC.
    Returns (F_ransac, inlier_mask) or (None, None) if not enough points.
    """
    if len(pts1) < 8:
        return None, None
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=3.0, confidence=0.99)
    if F is None or F.shape != (3, 3):
        return None, None
    return F, mask.ravel().astype(bool)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir",  type=str,
                   default="/work/hdd/bgkz/droid_raw_large_superset")
    p.add_argument("--raw-dir",      type=str, required=True)
    p.add_argument("--depth-dir",    type=str, required=True)
    p.add_argument("--canonical-id", type=str, required=True)
    p.add_argument("--cam2cam-json", type=str, default=None,
                   help="Path to cam2cam_extrinsics.json (annotation extrinsics). "
                        "When provided, output is 4-panel: "
                        "ext1 | metadata F | cam2cam F | RANSAC F")
    p.add_argument("--output",       type=str, default="epipolar.mp4")
    p.add_argument("--fps",          type=int, default=15)
    p.add_argument("--error-thresh", type=float, default=3.0,
                   help="Epipolar error threshold in pixels (green=good, red=bad)")
    p.add_argument("--max-matches",  type=int, default=20,
                   help="Max SIFT matches to draw per frame")
    return p.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    raw_dir     = Path(args.raw_dir)
    depth_dir   = Path(args.depth_dir)
    cid         = args.canonical_id

    # ── Extrinsics + intrinsics (metadata) ───────────────────────────────────
    meta        = load_meta(raw_dir, cid)
    ext1_serial = str(meta["ext1_cam_serial"])
    ext2_serial = str(meta["ext2_cam_serial"])

    T_ext1 = dof_to_mat(meta["ext1_cam_extrinsics"])
    T_ext2 = dof_to_mat(meta["ext2_cam_extrinsics"])

    intr_ext1 = np.load(depth_dir / cid / ext1_serial / "intrinsics.npy")
    intr_ext2 = np.load(depth_dir / cid / ext2_serial / "intrinsics.npy")

    K1_meta = intrinsics_matrix(intr_ext1)
    K2_meta = intrinsics_matrix(intr_ext2)

    F_meta = compute_fundamental(T_ext1, T_ext2, K1_meta, K2_meta)
    print(f"Metadata F:\n{np.round(F_meta, 6)}")
    sv = np.linalg.svd(F_meta, compute_uv=False)
    print(f"  rank={np.linalg.matrix_rank(F_meta)}  sv={np.round(sv, 6)}\n")

    # ── Extrinsics + intrinsics (cam2cam annotations, optional) ──────────────
    F_cam2cam = None
    if args.cam2cam_json is not None:
        T1_c, T2_c, K1_c, K2_c = load_cam2cam(Path(args.cam2cam_json), cid, raw_dir)
        F_cam2cam = compute_fundamental(T1_c, T2_c, K1_c, K2_c)
        print(f"Cam2cam annotation F:\n{np.round(F_cam2cam, 6)}")
        sv_c = np.linalg.svd(F_cam2cam, compute_uv=False)
        print(f"  rank={np.linalg.matrix_rank(F_cam2cam)}  sv={np.round(sv_c, 6)}\n")

    use_cam2cam = F_cam2cam is not None

    # ── Find mp4s ─────────────────────────────────────────────────────────────
    id_map_path = dataset_dir / "meta" / "episode_index_to_id.json"
    if not id_map_path.exists():
        raise ValueError(f"episode_index_to_id.json not found at {id_map_path}")
    with open(id_map_path) as f:
        id_map = json.load(f)
    reverse = {
        (v["canonical_id"] if isinstance(v, dict) else v): int(k)
        for k, v in id_map.items()
    }
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

    print(f"Decoding ext1 frames ...")
    frames_ext1 = decode_all_frames(mp4_ext1)
    print(f"Decoding ext2 frames ...")
    frames_ext2 = decode_all_frames(mp4_ext2)

    n_frames = min(len(frames_ext1), len(frames_ext2))
    print(f"Rendering {n_frames} frames ...\n")

    # Output writers — always 3-panel per video
    out_size = (_IMG_W * 3, _IMG_H)
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    out_stem = Path(args.output).with_suffix("")
    out_ext  = Path(args.output).suffix or ".mp4"

    if use_cam2cam:
        path_meta    = str(out_stem) + "_metadata" + out_ext
        path_cam2cam = str(out_stem) + "_cam2cam"  + out_ext
        writer_meta   = cv2.VideoWriter(path_meta,    fourcc, args.fps, out_size)
        writer_cam2cam = cv2.VideoWriter(path_cam2cam, fourcc, args.fps, out_size)
        print(f"Writing: {path_meta}")
        print(f"         {path_cam2cam}")
    else:
        path_meta  = args.output
        writer_meta = cv2.VideoWriter(path_meta, fourcc, args.fps, out_size)
        writer_cam2cam = None
        print(f"Writing: {path_meta}")

    all_errors_meta    = []
    all_errors_cam2cam = []
    all_errors_ransac  = []
    palette = [
        (255,   0,   0), (  0, 200,   0), (  0,   0, 255),
        (255, 165,   0), (128,   0, 128), (  0, 200, 200),
        (255, 105, 180), (210, 180, 140), ( 50, 205,  50),
        (255, 215,   0), (  0, 128, 255), (255,  20, 147),
        (  0, 255, 127), (220,  20,  60), ( 64, 224, 208),
        (148,   0, 211), (255, 140,   0), ( 34, 139,  34),
        (  0, 191, 255), (255,  99,  71),
    ]

    for fi in range(n_frames):
        img1 = cv2.resize(frames_ext1[fi], (_IMG_W, _IMG_H))
        img2 = cv2.resize(frames_ext2[fi], (_IMG_W, _IMG_H))
        bgr1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        bgr2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        pts1, pts2 = match_sift(gray1, gray2, max_matches=args.max_matches)

        # RANSAC F from matched points (data-driven, independent of our extrinsics)
        F_ransac, inlier_mask = ransac_fundamental(pts1, pts2)

        frame_errs_meta    = []
        frame_errs_cam2cam = []
        frame_errs_ransac  = []

        # Separate bgr1 panels per video so each shows its own per-match error label
        bgr1_meta    = bgr1.copy()
        bgr1_cam2cam = bgr1.copy() if use_cam2cam else None
        bgr2_meta    = bgr2.copy()
        bgr2_cam2cam = bgr2.copy() if use_cam2cam else None
        bgr2_ransac  = bgr2.copy() if F_ransac is not None else None

        for i, (p1, p2) in enumerate(zip(pts1, pts2)):
            color = palette[i % len(palette)]
            p1h = np.array([p1[0], p1[1], 1.0])
            p2h = np.array([p2[0], p2[1], 1.0])
            u1, v1 = int(round(p1[0])), int(round(p1[1]))
            u2, v2 = int(round(p2[0])), int(round(p2[1]))

            # --- metadata F ---
            err_m = epipolar_error(F_meta, p1h, p2h)
            frame_errs_meta.append(err_m)
            dot_m = (0, 220, 0) if err_m < args.error_thresh else (0, 0, 220)
            draw_epiline(bgr2_meta, epiline(F_meta, p1h), color)
            cv2.circle(bgr2_meta, (u2, v2), 5, dot_m, -1)
            cv2.circle(bgr2_meta, (u2, v2), 5, (255, 255, 255), 1)
            cv2.circle(bgr1_meta, (u1, v1), 5, color, -1)
            cv2.circle(bgr1_meta, (u1, v1), 5, (255, 255, 255), 1)
            cv2.putText(bgr1_meta, f"{err_m:.0f}", (u1 + 6, v1 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            # --- cam2cam annotation F ---
            if use_cam2cam and bgr2_cam2cam is not None:
                err_c = epipolar_error(F_cam2cam, p1h, p2h)
                frame_errs_cam2cam.append(err_c)
                dot_c = (0, 220, 0) if err_c < args.error_thresh else (0, 0, 220)
                draw_epiline(bgr2_cam2cam, epiline(F_cam2cam, p1h), color)
                cv2.circle(bgr2_cam2cam, (u2, v2), 5, dot_c, -1)
                cv2.circle(bgr2_cam2cam, (u2, v2), 5, (255, 255, 255), 1)
                cv2.circle(bgr1_cam2cam, (u1, v1), 5, color, -1)
                cv2.circle(bgr1_cam2cam, (u1, v1), 5, (255, 255, 255), 1)
                cv2.putText(bgr1_cam2cam, f"{err_c:.0f}", (u1 + 6, v1 + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            # --- RANSAC F ---
            if F_ransac is not None and bgr2_ransac is not None:
                err_r = epipolar_error(F_ransac, p1h, p2h)
                frame_errs_ransac.append(err_r)
                dot_r = (0, 220, 0) if err_r < args.error_thresh else (0, 0, 220)
                draw_epiline(bgr2_ransac, epiline(F_ransac, p1h), color)
                cv2.circle(bgr2_ransac, (u2, v2), 5, dot_r, -1)
                cv2.circle(bgr2_ransac, (u2, v2), 5, (255, 255, 255), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0

        if frame_errs_meta:
            all_errors_meta.extend(frame_errs_meta)
            label_meta = f"METADATA F: mean={np.mean(frame_errs_meta):.1f}px  n={len(frame_errs_meta)}"
        else:
            label_meta = "METADATA F: no matches"

        if frame_errs_cam2cam:
            all_errors_cam2cam.extend(frame_errs_cam2cam)
            label_c2c = f"CAM2CAM F: mean={np.mean(frame_errs_cam2cam):.1f}px  n={len(frame_errs_cam2cam)}"
        elif use_cam2cam:
            label_c2c = "CAM2CAM F: no matches"

        if frame_errs_ransac:
            all_errors_ransac.extend(frame_errs_ransac)
            label_r = f"RANSAC F: mean={np.mean(frame_errs_ransac):.1f}px  inliers={n_inliers}/{len(pts1)}"
        else:
            label_r = "RANSAC F: not computed"

        bgr2_ransac_panel = bgr2_ransac if bgr2_ransac is not None else bgr2.copy()

        # Metadata video: ext1 | metadata F | RANSAC F
        cv2.putText(bgr1_meta,        "ext1 (source pts)",  (5, 15), font, 0.4, (255,255,255), 1)
        cv2.putText(bgr2_meta,        "ext2 | metadata F",  (5, 15), font, 0.4, (255,255,255), 1)
        cv2.putText(bgr2_meta,        label_meta, (5, _IMG_H - 8), font, 0.33, (255,255,255), 1)
        cv2.putText(bgr2_ransac_panel,"ext2 | RANSAC F",    (5, 15), font, 0.4, (255,255,255), 1)
        cv2.putText(bgr2_ransac_panel, label_r,  (5, _IMG_H - 8), font, 0.33, (255,255,255), 1)
        writer_meta.write(np.concatenate([bgr1_meta, bgr2_meta, bgr2_ransac_panel], axis=1))

        # Cam2cam video: ext1 | cam2cam F | RANSAC F
        if use_cam2cam and bgr2_cam2cam is not None and writer_cam2cam is not None:
            bgr2_ransac_c2c = bgr2_ransac.copy() if bgr2_ransac is not None else bgr2.copy()
            cv2.putText(bgr1_cam2cam,    "ext1 (source pts)", (5, 15), font, 0.4, (255,255,255), 1)
            cv2.putText(bgr2_cam2cam,    "ext2 | cam2cam F",  (5, 15), font, 0.4, (255,255,255), 1)
            cv2.putText(bgr2_cam2cam,    label_c2c, (5, _IMG_H - 8), font, 0.33, (255,255,255), 1)
            cv2.putText(bgr2_ransac_c2c, "ext2 | RANSAC F",  (5, 15), font, 0.4, (255,255,255), 1)
            cv2.putText(bgr2_ransac_c2c, label_r,  (5, _IMG_H - 8), font, 0.33, (255,255,255), 1)
            writer_cam2cam.write(np.concatenate([bgr1_cam2cam, bgr2_cam2cam, bgr2_ransac_c2c], axis=1))

    writer_meta.release()
    if writer_cam2cam is not None:
        writer_cam2cam.release()

    sources = [("metadata F",  all_errors_meta),
               ("cam2cam F",  all_errors_cam2cam),
               ("RANSAC F",   all_errors_ransac)]

    print(f"\nEpipolar error summary over {n_frames} frames:")
    print(f"  {'':25s}  {'mean':>8}  {'median':>8}  {'p90':>8}  {'<thresh':>8}")
    for label, errs in sources:
        if not errs:
            print(f"  {label:25s}  (no data)")
            continue
        arr = np.array(errs)
        frac = np.mean(arr < args.error_thresh)
        print(f"  {label:25s}  {arr.mean():8.2f}  {np.median(arr):8.2f}  "
              f"{np.percentile(arr,90):8.2f}  {frac:8.1%}")
    print()
    print("Interpretation:")
    print("  If RANSAC F error is low but a source F error is high → that source is WRONG")
    print("  If both errors are high → SIFT matches are unreliable (not an extrinsics problem)")
    print("  If both errors are low  → extrinsics are CORRECT")
    print(f"\nSaved:")
    print(f"  {path_meta}")
    if use_cam2cam:
        print(f"  {path_cam2cam}")


if __name__ == "__main__":
    main()
