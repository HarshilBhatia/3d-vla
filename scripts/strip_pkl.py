"""
Strip unused fields from existing low_dim_obs.pkl files in-place.

Usage:
    python scripts/strip_pkl.py --root /grogu/user/harshilb/orbital_rollouts
    python scripts/strip_pkl.py --root /grogu/user/harshilb/orbital_rollouts --dry-run
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.processing.rlbench_utils import CustomUnpickler

_UNUSED_OBS_ATTRS = [
    "left_shoulder_rgb",    "left_shoulder_depth",    "left_shoulder_mask",    "left_shoulder_point_cloud",
    "right_shoulder_rgb",   "right_shoulder_depth",   "right_shoulder_mask",   "right_shoulder_point_cloud",
    "overhead_rgb",         "overhead_depth",          "overhead_mask",         "overhead_point_cloud",
    "front_rgb",            "front_depth",             "front_mask",            "front_point_cloud",
    "wrist_mask",           "wrist_point_cloud",
    "joint_forces",         "gripper_matrix",          "gripper_joint_positions",
    "gripper_touch_forces", "task_low_dim_state",      "ignore_collisions",
    "mesh_points",
]

_WRIST_MISC_KEYS = {
    "wrist_camera_near", "wrist_camera_far",
    "wrist_camera_extrinsics", "wrist_camera_intrinsics",
}


def _strip_obs(obs):
    for attr in _UNUSED_OBS_ATTRS:
        if hasattr(obs, attr):
            setattr(obs, attr, None)
    if hasattr(obs, "misc") and isinstance(obs.misc, dict):
        obs.misc = {k: v for k, v in obs.misc.items() if k in _WRIST_MISC_KEYS}


def sizeof_fmt(num):
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def strip_file(path, dry_run):
    before = os.path.getsize(path)
    try:
        with open(path, "rb") as f:
            demo = CustomUnpickler(f).load()
    except Exception as e:
        print(f"  [SKIP] could not load {path}: {e}")
        return 0, 0

    observations = getattr(demo, "_observations", None)
    if observations is None:
        print(f"  [SKIP] no _observations in {path}")
        return 0, 0

    for obs in observations:
        _strip_obs(obs)

    if not dry_run:
        with open(path, "wb") as f:
            pickle.dump(demo, f)

    after = os.path.getsize(path) if not dry_run else None
    return before, after


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Root dir to search for low_dim_obs.pkl")
    p.add_argument("--dry-run", action="store_true", help="Report sizes without writing")
    args = p.parse_args()

    pkls = []
    for dirpath, _, filenames in os.walk(args.root):
        if "low_dim_obs.pkl" in filenames:
            pkls.append(os.path.join(dirpath, "low_dim_obs.pkl"))

    pkls.sort()
    print(f"Found {len(pkls)} pkl files under {args.root}")
    if args.dry_run:
        print("DRY RUN — no files will be modified\n")

    total_before = total_after = 0
    for path in pkls:
        before, after = strip_file(path, args.dry_run)
        if before == 0:
            continue
        total_before += before
        if not args.dry_run:
            total_after += after
            saved = before - after
            print(f"  {path}\n    {sizeof_fmt(before)} -> {sizeof_fmt(after)}  (saved {sizeof_fmt(saved)})")
        else:
            total_after += before  # unchanged in dry-run
            print(f"  {path}  ({sizeof_fmt(before)})")

    if not args.dry_run:
        print(f"\nTotal: {sizeof_fmt(total_before)} -> {sizeof_fmt(total_after)}  "
              f"(saved {sizeof_fmt(total_before - total_after)})")
    else:
        print(f"\nTotal size: {sizeof_fmt(total_before)}")


if __name__ == "__main__":
    main()
