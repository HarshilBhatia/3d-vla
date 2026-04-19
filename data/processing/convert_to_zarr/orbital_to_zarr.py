"""
Convert raw orbital rollout episodes → single train.zarr.

Directory layout expected (from collect_orbital_rollouts.py):
  {root}/{task}/{group}/episode_{N}/
      orbital_left_rgb/   {0000..}.png
      orbital_left_depth/ {0000..}.png   (RGB-encoded float, RLBench convention)
      orbital_right_rgb/  {0000..}.png
      orbital_right_depth/{0000..}.png
      over_shoulder_left_rgb/   {0000..}.png
      over_shoulder_left_depth/ {0000..}.png
      over_shoulder_right_rgb/  {0000..}.png
      over_shoulder_right_depth/{0000..}.png
      low_dim_obs.pkl
      camera_group.txt

Zarr schema (matches Peract2_zarr / peract2_to_zarr.py):
  rgb              (N, NCAM=4, 3, H, W)   uint8
  depth            (N, NCAM=4, H, W)      float16
  extrinsics       (N, NCAM=4, 4, 4)      float16  cam-to-world
  intrinsics       (N, NCAM=4, 3, 3)      float16
  proprioception   (N, 3, NHAND=1, 8)     float32
  action           (N, 1, NHAND=1, 8)     float32
  proprioception_joints (N, 1, NHAND=1, 8) float32
  action_joints    (N, 1, NHAND=1, 8)     float32
  task_id          (N,)                   uint8
  variation        (N,)                   uint8
  camera_group     (N,)                   uint8   (1-6)

Camera order:  [orbital_left, orbital_right, wrist]
"""
import argparse
import json
import os
import sys

import numpy as np
from numcodecs import Blosc
from tqdm import tqdm
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from data.processing.rlbench_utils import PERACT_TASKS
from data.processing.orbital_utils import process_episode

NCAM   = 3
NHAND  = 1
IM_SIZE = 256


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert orbital rollouts to train.zarr"
    )
    p.add_argument("--root",       required=True,
                   help="Root dir containing task/group/episode_* folders")
    p.add_argument("--out",        required=True,
                   help="Output zarr path (e.g. data/orbital_train.zarr)")
    p.add_argument("--image-size", type=int, default=IM_SIZE)
    p.add_argument("--tasks",      default=None,
                   help="Comma-separated task list (default: all 18 PerAct)")
    p.add_argument("--groups",     default=None,
                   help="Comma-separated camera groups to include (e.g. G2,G3). Default: all groups present.")
    p.add_argument("--overwrite",  action="store_true",
                   help="Remove existing zarr and rebuild")
    return p.parse_args()


def main():
    args = parse_args()

    tasks = PERACT_TASKS
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
    task2id = {t: i for i, t in enumerate(PERACT_TASKS)}
    allowed_groups = None
    if args.groups:
        allowed_groups = set(g.strip() for g in args.groups.split(","))

    if os.path.exists(args.out):
        if args.overwrite:
            import shutil
            shutil.rmtree(args.out)
            print("[INFO] Removed existing zarr at {}".format(args.out))
        else:
            print("[SKIP] {} already exists. Use --overwrite to rebuild.".format(
                args.out))
            return

    im = args.image_size
    compressor = Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)

    with zarr.open_group(args.out, mode="w") as zf:

        def _create(name, shape, dtype):
            zf.create_dataset(
                name, shape=(0,) + shape,
                chunks=(1,) + shape,
                compressor=compressor, dtype=dtype,
            )

        _create("rgb",                   (NCAM, 3, im, im), "uint8")
        _create("depth",                 (NCAM, im, im),    "float16")
        _create("extrinsics",            (NCAM, 4, 4),      "float16")
        _create("intrinsics",            (NCAM, 3, 3),      "float16")
        _create("proprioception",        (3, NHAND, 8),     "float32")
        _create("action",                (1, NHAND, 8),     "float32")
        _create("proprioception_joints", (1, NHAND, 8),     "float32")
        _create("action_joints",         (1, NHAND, 8),     "float32")
        _create("task_id",               (),                "uint8")
        _create("variation",             (),                "uint8")
        _create("camera_group",          (),                "uint8")

        total = 0
        for task in tasks:
            tid = task2id.get(task, 0)
            task_root = os.path.join(args.root, task)
            if not os.path.isdir(task_root):
                print("[SKIP] No data for task {}".format(task))
                continue

            groups = sorted(os.listdir(task_root))
            for group_str in groups:
                if allowed_groups is not None and group_str not in allowed_groups:
                    continue
                group_root = os.path.join(task_root, group_str)
                if not os.path.isdir(group_root):
                    continue
                episodes = sorted([
                    d for d in os.listdir(group_root)
                    if d.startswith("episode_") and
                       os.path.isdir(os.path.join(group_root, d))
                ])
                print("[{}] {} — {} episodes".format(
                    task, group_str, len(episodes)))
                for ep in tqdm(episodes, desc="{}/{}".format(task, group_str)):
                    ep_path = os.path.join(group_root, ep)
                    n = process_episode(ep_path, tid, group_str, zf, im)
                    total += n

        print("\n[DONE] Wrote {} keyframe rows to {}".format(total, args.out))
        for key in zf.keys():
            print("  {}: {}".format(key, zf[key].shape))


if __name__ == "__main__":
    main()
