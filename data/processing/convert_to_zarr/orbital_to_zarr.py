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

Zarr schema:
  rgb              (N, NCAM=3, 3, H, W)   uint8
  depth            (N, NCAM=3, H, W)      float16  metric depth in metres
  extrinsics       (N, NCAM=3, 4, 4)      float16  cam-to-world
  intrinsics       (N, NCAM=3, 3, 3)      float16
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
                   help="Comma-separated task list (default: profile task list)")
    p.add_argument("--groups",     default=None,
                   help="Comma-separated camera groups to include (e.g. G2,G3). Default: all groups present.")
    p.add_argument("--overwrite",  action="store_true",
                   help="Remove existing zarr and rebuild")
    p.add_argument("--bimanual",   action="store_true",
                   help="Use PerAct2 (dual_panda) profile instead of PerAct (panda)")
    p.add_argument("--store-trajectory", action="store_true",
                   help="Store dense interpolated trajectory instead of next keypose only")
    p.add_argument("--interp-len", type=int, default=50,
                   help="Number of interpolated steps per keypose segment (default: 50)")
    return p.parse_args()


def main():
    args = parse_args()

    from data.generation.orbital.constants import PERACT_PROFILE, PERACT2_PROFILE
    profile = PERACT2_PROFILE if args.bimanual else PERACT_PROFILE

    ncam  = 2 + len(profile.wrist_cameras)  # orbital_left + orbital_right + wrist cam(s)
    nhand = profile.nhand

    tasks = profile.task_list
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
    task2id = {t: i for i, t in enumerate(profile.task_list)}
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

        _create("rgb",                   (ncam, 3, im, im), "uint8")
        _create("depth",                 (ncam, im, im),    "float16")
        _create("extrinsics",            (ncam, 4, 4),      "float16")
        _create("intrinsics",            (ncam, 3, 3),      "float16")
        _create("proprioception",        (3, nhand, 8),     "float32")
        action_len = args.interp_len if args.store_trajectory else 1
        _create("action",                (action_len, nhand, 8), "float32")
        _create("proprioception_joints", (1, nhand, 8),     "float32")
        _create("action_joints",         (1, nhand, 8),     "float32")
        _create("task_id",               (),                "uint8")
        _create("variation",             (),                "uint8")
        _create("camera_group",          (),                "uint8")
        _create("demo_id",               (),                "uint32")

        total = 0
        n_episodes = 0
        for task in tasks:
            tid = task2id.get(task, 0)
            task_root = os.path.join(args.root, task)
            if not os.path.isdir(task_root):
                print(task_root)
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
                    try:
                        n = process_episode(ep_path, tid, group_str, zf, im, demo_id=n_episodes, profile=profile,
                                            store_trajectory=args.store_trajectory, interp_len=args.interp_len)
                        total += n
                        n_episodes += 1
                    except Exception as e:
                        print("[WARN] Skipping {}: {}".format(ep_path, e))

        mode_str = "dense (interp_len={})".format(args.interp_len) if args.store_trajectory else "sparse (keypose-only)"
        print("\n[DONE] Wrote {} keyframe rows ({}) to {}".format(total, mode_str, args.out))
        for key in zf.keys():
            print("  {}: {}".format(key, zf[key].shape))


if __name__ == "__main__":
    main()
