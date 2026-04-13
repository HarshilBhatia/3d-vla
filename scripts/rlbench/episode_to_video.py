"""Convert a RLBench episode directory to a tiled MP4.

Layout:
  [ left_shoulder | front | right_shoulder ]
  [   overhead    | wrist |    (black)     ]

Usage:
  python scripts/rlbench/episode_to_video.py \
      --episode_dir peract_raw_rot10/place_cups/all_variations/episodes/episode0 \
      --output /tmp/episode0.mp4 \
      --fps 10
"""

import argparse
import os
import numpy as np
from PIL import Image
import imageio

CAMERAS = [
    "left_shoulder_rgb",
    "front_rgb",
    "right_shoulder_rgb",
    "overhead_rgb",
    "wrist_rgb",
]

LAYOUT = [
    ["left_shoulder_rgb", "front_rgb",   "right_shoulder_rgb"],
    ["overhead_rgb",      "wrist_rgb",   None],
]


def load_frames(cam_dir):
    files = [f for f in os.listdir(cam_dir) if f.endswith(".png")]
    files.sort(key=lambda f: int(f.replace(".png", "")))
    return [np.array(Image.open(os.path.join(cam_dir, f)).convert("RGB")) for f in files]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="Render only these cameras as a horizontal strip")
    args = parser.parse_args()

    ep = args.episode_dir.rstrip("/")
    out = args.output or os.path.join(ep, "episode.mp4")

    available = [c for c in CAMERAS if os.path.isdir(os.path.join(ep, c))]
    if not available:
        raise RuntimeError(f"No camera folders found in {ep}")

    cams = [c for c in (args.cameras or available) if c in available]
    print(f"Cameras: {cams}")

    all_frames = {}
    for c in cams:
        frames = load_frames(os.path.join(ep, c))
        all_frames[c] = frames
        print(f"  {c}: {len(frames)} frames")

    n = min(len(v) for v in all_frames.values())
    h, w = all_frames[cams[0]][0].shape[:2]
    black = np.zeros((h, w, 3), dtype=np.uint8)
    print(f"Writing {n} frames @ {args.fps} fps -> {out}")

    if args.cameras:
        layout = [cams]
    else:
        layout = [[c for c in row] for row in LAYOUT]

    cols = max(len(r) for r in layout)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    writer = imageio.get_writer(out, fps=args.fps, codec="libx264",
                                quality=8, pixelformat="yuv420p")
    for fi in range(n):
        rows = []
        for row in layout:
            row_imgs = []
            for c in row:
                row_imgs.append(all_frames[c][fi] if (c and c in all_frames) else black)
            while len(row_imgs) < cols:
                row_imgs.append(black)
            rows.append(np.concatenate(row_imgs, axis=1))
        writer.append_data(np.concatenate(rows, axis=0))
    writer.close()
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
