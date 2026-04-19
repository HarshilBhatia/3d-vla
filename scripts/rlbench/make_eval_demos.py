"""
Create PerAct-format eval demo directories from orbital rollout data.

The online evaluator (evaluate_policy.py) expects:
  {out}/{task}/variation{V}/episodes/episode{N}/low_dim_obs.pkl

We have:
  {root}/{task}/{group}/episode_{N}/low_dim_obs.pkl

All rollouts use variation 0, so we flatten all groups into a single
variation0 pool and symlink the low_dim_obs.pkl files.

Usage:
  python scripts/rlbench/make_eval_demos.py \
      --root data/orbital_rollouts \
      --out  data/eval_demos

  # Specific tasks only:
  python scripts/rlbench/make_eval_demos.py \
      --root data/orbital_rollouts \
      --out  data/eval_demos \
      --tasks place_cups,close_jar

  # Copy instead of symlink (needed if moving across filesystems):
      --copy
"""

import argparse
import os
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root",      required=True,
                   help="Root of orbital_rollouts (task/group/episode_* dirs)")
    p.add_argument("--out",       required=True,
                   help="Output root for eval demos")
    p.add_argument("--tasks",     default=None,
                   help="Comma-separated task list (default: all tasks in root)")
    p.add_argument("--variation", type=int, default=0,
                   help="Variation number to place demos under (default: 0)")
    p.add_argument("--copy",      action="store_true",
                   help="Copy files instead of symlinking")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    out  = Path(args.out)

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
    else:
        tasks = sorted(p.name for p in root.iterdir() if p.is_dir())

    total = 0
    for task in tasks:
        task_root = root / task
        if not task_root.is_dir():
            print(f"[SKIP] {task} — not found in {root}")
            continue

        # Collect all low_dim_obs.pkl across all groups, sorted deterministically
        src_pkls = sorted(task_root.glob("G*/episode_*/low_dim_obs.pkl"))
        if not src_pkls:
            print(f"[SKIP] {task} — no episodes found")
            continue

        ep_out_root = out / task / f"variation{args.variation}" / "episodes"
        ep_out_root.mkdir(parents=True, exist_ok=True)

        for ep_idx, src in enumerate(src_pkls):
            ep_dir = ep_out_root / f"episode{ep_idx}"
            ep_dir.mkdir(exist_ok=True)
            dst = ep_dir / "low_dim_obs.pkl"

            if dst.exists() or dst.is_symlink():
                dst.unlink()

            if args.copy:
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src.resolve())

        print(f"[OK] {task}/variation{args.variation} — {len(src_pkls)} episodes")
        total += len(src_pkls)

    print(f"\n[DONE] {total} episodes → {out}")


if __name__ == "__main__":
    main()
