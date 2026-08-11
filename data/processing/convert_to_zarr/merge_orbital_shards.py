"""
Merge per-shard orbital zarrs into a single train.zarr + val.zarr.

The orbital collection sweep runs one (task, camera_group) shard per process and
converts each shard independently (orbital_to_zarr.py writes shards/{task}__{group}/
{train,val}.zarr). Raw episodes are ~230 MB each, so they are deleted as soon as
their shard is converted; this script stitches the surviving shard zarrs together.

demo_id is rebased per shard so it stays unique across the merged output.

--mapping restricts the merge to each task's designated train_groups (see
instructions/peract2_orbital_task_group_mapping.json). Shards outside that
assignment -- including every task's held-out eval_group -- stay on disk but
are excluded from the merged zarrs.

Usage:
    python data/processing/convert_to_zarr/merge_orbital_shards.py \
        --shards /path/to/shards --out /path/to/zarr \
        --mapping instructions/peract2_orbital_task_group_mapping.json
"""
import argparse
import json
import os
import shutil

import numpy as np
from numcodecs import Blosc
from tqdm import tqdm
import zarr

# Row-aligned arrays copied verbatim from every shard.
KEYS = [
    "rgb", "depth", "extrinsics", "intrinsics",
    "proprioception", "action", "proprioception_joints", "action_joints",
    "task_id", "variation", "camera_group", "demo_id",
]


def parse_args():
    p = argparse.ArgumentParser(description="Merge per-shard orbital zarrs.")
    p.add_argument("--shards", required=True,
                   help="Directory containing {task}__{group}/{train,val}.zarr")
    p.add_argument("--out", required=True,
                   help="Output directory for the merged train.zarr / val.zarr")
    p.add_argument("--overwrite", action="store_true",
                   help="Remove existing merged zarrs first")
    p.add_argument("--batch-rows", type=int, default=64,
                   help="Rows copied per append (bounds peak memory)")
    p.add_argument("--mapping",
                   help="Task -> camera-group mapping JSON. When given, only "
                        "each task's train_groups shards are merged.")
    return p.parse_args()


def load_allowed_shards(path):
    """Return the set of '{task}__{group}' shard names a mapping allows."""
    with open(path) as f:
        doc = json.load(f)
    allowed = set()
    for task, spec in doc["tasks"].items():
        for group in spec["train_groups"]:
            allowed.add("{}__{}".format(task, group))
    return allowed


def _init_like(path, src, compressor):
    """Create an empty zarr group with the same per-row shapes/dtypes as src."""
    zf = zarr.open_group(path, mode="w")
    for key in KEYS:
        row_shape = src[key].shape[1:]
        zf.create_dataset(
            key, shape=(0,) + row_shape, chunks=(1,) + row_shape,
            compressor=compressor, dtype=src[key].dtype,
        )
    return zf


def _copy_shard(src, dst, demo_offset, batch_rows):
    """Append all rows of src to dst, rebasing demo_id by demo_offset.

    Returns (n_rows, n_demos) contributed by this shard.
    """
    n = int(src["rgb"].shape[0])
    if n == 0:
        return 0, 0
    for start in range(0, n, batch_rows):
        stop = min(start + batch_rows, n)
        for key in KEYS:
            block = src[key][start:stop]
            if key == "demo_id":
                block = block.astype(np.uint32) + demo_offset
            dst[key].append(block)
    n_demos = int(src["demo_id"][:].max()) + 1
    return n, n_demos


def main():
    args = parse_args()

    shard_names = sorted(
        d for d in os.listdir(args.shards)
        if os.path.isdir(os.path.join(args.shards, d))
    )
    complete = [
        d for d in shard_names
        if os.path.exists(os.path.join(args.shards, d, ".complete"))
    ]
    skipped = sorted(set(shard_names) - set(complete))
    if skipped:
        print("[WARN] Ignoring {} incomplete shard(s): {}".format(
            len(skipped), ", ".join(skipped)))

    if args.mapping:
        allowed = load_allowed_shards(args.mapping)
        selected = [d for d in complete if d in allowed]
        excluded = sorted(set(complete) - allowed)
        missing = sorted(allowed - set(complete))
        print("[INFO] Mapping {} selects {} shard(s)".format(
            args.mapping, len(allowed)))
        print("[INFO] Excluded {} collected shard(s) (kept on disk)".format(
            len(excluded)))
        if missing:
            print("[WARN] {} mapped shard(s) not collected: {}".format(
                len(missing), ", ".join(missing)))
        complete = selected

    print("[INFO] Merging {} complete shard(s)".format(len(complete)))

    os.makedirs(args.out, exist_ok=True)
    compressor = Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)

    out_paths = {s: os.path.join(args.out, s + ".zarr") for s in ("train", "val")}
    if args.overwrite:
        for path in out_paths.values():
            if os.path.exists(path):
                shutil.rmtree(path)
                print("[INFO] Removed existing {}".format(path))

    out_zf = {}
    totals = {s: [0, 0] for s in out_paths}  # split -> [rows, demos]

    for shard in tqdm(complete, desc="shards"):
        for split, out_path in out_paths.items():
            src_path = os.path.join(args.shards, shard, split + ".zarr")
            if not os.path.exists(src_path):
                continue
            src = zarr.open_group(src_path, mode="r")
            if split not in out_zf:
                out_zf[split] = _init_like(out_path, src, compressor)
            rows, demos = _copy_shard(
                src, out_zf[split], totals[split][1], args.batch_rows)
            totals[split][0] += rows
            totals[split][1] += demos

    for split, zf in out_zf.items():
        rows, demos = totals[split]
        print("\n[DONE] {}.zarr — {} rows, {} episodes".format(split, rows, demos))
        for key in KEYS:
            print("    {}: {}".format(key, zf[key].shape))


if __name__ == "__main__":
    main()
