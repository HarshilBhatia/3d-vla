"""
Render frames from a PerAct- or PerAct2-style zarr to a video file for verification.

Zarr layout: (total_frames, NCAM, 3, H, W). Use --max_frames to limit (default: 60).
PerAct has 4 cameras (left_shoulder, right_shoulder, wrist, front); PerAct2 has 3 (front, wrist_left, wrist_right).

Usage:
  python data/processing/zarr_to_video.py --zarr /path/to/val.zarr --out verify.mp4 [--camera front] [--max_frames 60]
"""
import argparse
import os
import sys

import numpy as np
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
ZARR_ROOT = "Peract2_zarr"

# PerAct (4 cams) vs PerAct2 (3 cams)
PERACT_CAMERAS = ["left_shoulder", "right_shoulder", "wrist", "front"]
PERACT2_CAMERAS = ["front", "wrist_left", "wrist_right"]


def parse_args():
    p = argparse.ArgumentParser(description="Render zarr frames to video for verification")
    p.add_argument("--zarr", type=str, default=None, help="Path to .zarr (e.g. val.zarr or train.zarr)")
    p.add_argument("--out", type=str, default="zarr_verify.mp4", help="Output video path")
    p.add_argument("--camera", type=str, default="front", help="Camera name (front, wrist, etc.)")
    p.add_argument("--max_frames", type=int, default=60, help="Max frames to render (from start)")
    p.add_argument("--fps", type=int, default=5, help="Frames per second")
    return p.parse_args()


def main():
    args = parse_args()
    zarr_path = args.zarr
    if zarr_path is None:
        zarr_path = os.path.join(ZARR_ROOT, "val.zarr")
    if not zarr_path.endswith(".zarr"):
        zarr_path = zarr_path.rstrip("/") + ".zarr"
    if not os.path.isdir(zarr_path):
        print(f"[ERROR] Zarr not found: {zarr_path}")
        sys.exit(1)

    group = zarr.open_group(zarr_path, mode="r")
    if "rgb" not in group:
        print("[ERROR] No 'rgb' in zarr")
        sys.exit(1)
    rgb = group["rgb"]
    # Shape: (total_frames, NCAM, 3, H, W)
    total = rgb.shape[0]
    n_cams = rgb.shape[1]
    if total == 0:
        print(f"[ERROR] Zarr has 0 frames: {zarr_path}. Build the zarr from raw data first.")
        sys.exit(1)
    n_frames = min(args.max_frames, total)
    if n_cams == 4:
        cameras = PERACT_CAMERAS
    else:
        cameras = PERACT2_CAMERAS
    if args.camera not in cameras:
        print(f"[ERROR] Camera {args.camera!r} not in {cameras}")
        sys.exit(1)
    cam_idx = cameras.index(args.camera)
    frames = rgb[:n_frames, cam_idx]  # (n_frames, 3, H, W)
    # (T, H, W, 3) for video
    frames = np.transpose(frames, (0, 2, 3, 1))

    if frames.shape[0] == 0:
        print(f"[ERROR] No frames to write. Zarr has {total} total.")
        sys.exit(1)

    try:
        import imageio
        imageio.mimsave(args.out, frames, fps=args.fps)
    except ImportError:
        try:
            import cv2
            h, w = frames.shape[1], frames.shape[2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))
            for i in range(frames.shape[0]):
                out.write(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
            out.release()
        except ImportError:
            print("[ERROR] Install imageio or opencv-python to write video")
            sys.exit(1)

    print(f"[OK] Wrote {args.out} (camera={args.camera}, {n_frames} frames, {total} total in zarr)")


if __name__ == "__main__":
    main()
