"""
build_episode_frame_index.py

Builds two lookup files needed for on-the-fly depth loading during training:

1. episode_frame_index.pt
   Maps each cache global_idx to its (canonical_id, frame_idx).
   Format: list of dicts [{"canonical_id": str, "frame_idx": int}, ...]
   Reconstructed deterministically from the same ShardedSingleStepDataset
   params used when caching backbone features (seed=42).

2. serial_map.json
   Maps canonical_id → {"ext1": serial_str, "wrist": serial_str}
   Built by scanning metadata_*.json files in the raw RAIL download dir.

Usage:
    python data_processing/build_episode_frame_index.py \
        --dataset-path /work/nvme/bgkz/droid_raw_large_superset \
        --raw-dir /work/nvme/bgkz/droid_rail_raw \
        --output-dir /work/nvme/bgkz/droid_rail_depths \
        --labs RAIL \
        --shard-size 10000 \
        --episode-sampling-rate 0.1

The output files are saved to --output-dir.
Both files are required by Gr00tN1d6DataCollator when depth_dir is set.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch

from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS


def get_allowed_episode_indices(dataset_path: str, labs: list[str]) -> set[int]:
    id_map_path = Path(dataset_path) / "meta" / "episode_index_to_id.json"
    with open(id_map_path) as f:
        id_map = json.load(f)
    allowed = set()
    for idx_str, meta in id_map.items():
        lab = meta["canonical_id"].split("+")[0]
        if lab in labs:
            allowed.add(int(idx_str))
    return allowed


def build_episode_frame_index(base_dataset, dataset_path: str) -> list[dict]:
    """Walk sharded_episodes to build global_idx → (canonical_id, frame_idx)."""
    id_map_path = Path(dataset_path) / "meta" / "episode_index_to_id.json"
    with open(id_map_path) as f:
        id_map = json.load(f)

    index = []
    for shard_idx in range(len(base_dataset)):
        for ep_idx, step_indices in base_dataset.sharded_episodes[shard_idx]:
            canonical_id = id_map[str(ep_idx)]["canonical_id"]
            for step_index in step_indices:
                index.append({
                    "canonical_id": canonical_id,
                    "frame_idx": int(step_index),
                })
    return index


def build_serial_map(raw_dir: Path, canonical_ids: set[str]) -> dict:
    """Build canonical_id → {ext1: serial, wrist: serial} from raw metadata JSONs."""
    serial_map = {}
    for ep_dir in raw_dir.iterdir():
        if not ep_dir.is_dir():
            continue
        canonical_id = ep_dir.name
        if canonical_id not in canonical_ids:
            continue
        meta_files = list(ep_dir.glob("metadata_*.json"))
        if not meta_files:
            print(f"  [WARN] No metadata for {canonical_id}", file=sys.stderr)
            continue
        meta = json.loads(meta_files[0].read_text())
        serial_map[canonical_id] = {
            "ext1":  str(meta["ext1_cam_serial"]),
            "wrist": str(meta["wrist_cam_serial"]),
        }
    return serial_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--raw-dir", required=True, type=Path,
                        help="Directory with raw RAIL downloads (metadata_*.json files)")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Where to write episode_frame_index.pt and serial_map.json")
    parser.add_argument("--labs", nargs="+", default=["RAIL"])
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--episode-sampling-rate", type=float, default=0.1)
    parser.add_argument("--embodiment-tag", default="OXE_DROID")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    embodiment_tag = EmbodimentTag[args.embodiment_tag]
    modality_config = MODALITY_CONFIGS[args.embodiment_tag.lower()]

    allowed_episode_indices = None
    if args.labs:
        allowed_episode_indices = get_allowed_episode_indices(args.dataset_path, args.labs)
        print(f"Lab filter {args.labs}: {len(allowed_episode_indices)} episodes")

    print("Building ShardedSingleStepDataset (same params as backbone cache)...")
    base_dataset = ShardedSingleStepDataset(
        dataset_path=args.dataset_path,
        embodiment_tag=embodiment_tag,
        modality_configs=modality_config,
        shard_size=args.shard_size,
        episode_sampling_rate=args.episode_sampling_rate,
        seed=42,
        allow_padding=False,
        allowed_episode_indices=allowed_episode_indices,
    )
    total_samples = sum(base_dataset.get_shard_length(i) for i in range(len(base_dataset)))
    print(f"Dataset: {len(base_dataset)} shards, {total_samples} samples")

    # Build episode_frame_index
    print("Building episode_frame_index...")
    index = build_episode_frame_index(base_dataset, args.dataset_path)
    assert len(index) == total_samples, f"Index length mismatch: {len(index)} vs {total_samples}"

    out_index = args.output_dir / "episode_frame_index.pkl"
    with open(out_index, "wb") as f:
        pickle.dump(index, f, protocol=4)
    print(f"Saved {len(index)} entries to {out_index}")

    # Build serial_map
    canonical_ids = set(entry["canonical_id"] for entry in index)
    print(f"Building serial_map for {len(canonical_ids)} episodes...")
    serial_map = build_serial_map(args.raw_dir, canonical_ids)
    print(f"  Found metadata for {len(serial_map)}/{len(canonical_ids)} episodes")

    out_serial = args.output_dir / "serial_map.json"
    with open(out_serial, "w") as f:
        json.dump(serial_map, f, indent=2)
    print(f"Saved serial_map to {out_serial}")


if __name__ == "__main__":
    main()
