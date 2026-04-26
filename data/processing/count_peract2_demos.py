"""
Count PerAct2 demonstrations (episodes) from raw data or from zarr.

Usage:
  python -m data.processing.count_peract2_demos --root peract2_raw
  python -m data.processing.count_peract2_demos --zarr Peract2_zarr/val.zarr
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
RAW_ROOT = "peract2_raw"
ZARR_ROOT = "Peract2_zarr"

from data.processing.convert_to_zarr.peract2_to_zarr import DEFAULT_TASKS


def count_from_raw(root: str, tasks: list) -> dict:
    """Count episodes per task and total from raw folder structure."""
    counts = {}
    total = 0
    for task in tasks:
        for split in ("train", "val"):
            for folder in [
                f"{root}/{split}/{task}/all_variations/episodes",
                f"{root}/{task}/all_variations/episodes",
            ]:
                if os.path.isdir(folder):
                    n = len([e for e in os.listdir(folder) if e.startswith("ep")])
                    key = f"{task}/{split}"
                    counts[key] = counts.get(key, 0) + n
                    total += n
                    break
            else:
                task_base = f"{root}/{task}"
                if not os.path.isdir(task_base):
                    task_base = f"{root}/{split}/{task}"
                if os.path.isdir(task_base):
                    n = 0
                    for v in sorted(os.listdir(task_base)):
                        if v.startswith("variation"):
                            ep_dir = os.path.join(task_base, v, "episodes")
                            if os.path.isdir(ep_dir):
                                n += len([e for e in os.listdir(ep_dir) if e.startswith("ep")])
                    if n > 0:
                        key = f"{task}/{split}"
                        counts[key] = counts.get(key, 0) + n
                        total += n
    return {"by_task_split": counts, "total": total}


def count_from_zarr(zarr_path: str) -> int:
    """Count total frames in a single zarr (concatenated layout has no rollout boundary)."""
    import zarr
    g = zarr.open_group(zarr_path, mode="r")
    if "rgb" in g:
        return g["rgb"].shape[0]
    if "task_id" in g:
        return g["task_id"].shape[0]
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None, help="Raw data root (e.g. peract2_raw)")
    ap.add_argument("--zarr", type=str, default=None, help="Path to one .zarr (e.g. val.zarr)")
    ap.add_argument("--tasks", type=str, default=None, help="Comma-separated tasks; default: all PerAct2")
    args = ap.parse_args()
    tasks = args.tasks.split(",") if args.tasks else DEFAULT_TASKS
    if args.zarr:
        path = args.zarr if args.zarr.endswith(".zarr") else args.zarr.rstrip("/") + ".zarr"
        if not os.path.isdir(path):
            print(f"Zarr not found: {path}", file=sys.stderr)
            sys.exit(1)
        n = count_from_zarr(path)
        print(f"Total frames in zarr (concatenated layout): {n}")
        return
    root = args.root or RAW_ROOT
    if not os.path.isdir(root):
        print(f"Root not found: {root}", file=sys.stderr)
        sys.exit(1)
    out = count_from_raw(root, tasks)
    print("PerAct2 demonstrations (by task/split):")
    for k, v in sorted(out["by_task_split"].items()):
        if v:
            print(f"  {k}: {v}")
    print(f"Total episodes: {out['total']}")


if __name__ == "__main__":
    main()
