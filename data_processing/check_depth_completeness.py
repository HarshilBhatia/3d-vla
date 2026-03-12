"""
check_depth_completeness.py

Validates depth data completeness for all RAIL episodes before training.
For each episode, checks:
  1. ext1 depth.blosc + .done sentinel exists
  2. wrist depth.blosc + .done sentinel exists
  3. raw metadata_*.json exists (for ext1 static extrinsics)
  4. trajectory.h5 exists and contains wrist camera extrinsics key

Writes:
  {output_dir}/valid_canonical_ids.json   — list of fully-complete episodes
  {output_dir}/invalid_episodes.json      — dict of {canonical_id: [reasons]}

Usage:
    python data_processing/check_depth_completeness.py \
        --depth-dir /work/nvme/bgkz/droid_rail_depths \
        --raw-dir   /work/nvme/bgkz/droid_rail_raw \
        --serial-map /work/nvme/bgkz/droid_rail_depths/serial_map.json \
        [--output-dir /work/nvme/bgkz/droid_rail_depths]
"""

import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def check_episode(canonical_id: str, depth_dir: Path, raw_dir: Path, serial_map: dict) -> list[str]:
    """Return list of failure reasons; empty list means episode is valid."""
    issues = []

    serials = serial_map.get(canonical_id)
    if serials is None:
        return ["no serial map entry"]

    ext1_serial = serials["ext1"]
    wrist_serial = serials["wrist"]

    # 1. ext1 depth
    ext1_dir = depth_dir / canonical_id / ext1_serial
    if not (ext1_dir / ".done").exists():
        issues.append(f"ext1 .done missing ({ext1_serial})")
    elif not (ext1_dir / "depth.blosc").exists():
        issues.append(f"ext1 depth.blosc missing ({ext1_serial})")

    # 2. wrist depth
    wrist_dir = depth_dir / canonical_id / wrist_serial
    if not (wrist_dir / ".done").exists():
        issues.append(f"wrist .done missing ({wrist_serial})")
    elif not (wrist_dir / "depth.blosc").exists():
        issues.append(f"wrist depth.blosc missing ({wrist_serial})")

    # 3. raw metadata (for ext1 static extrinsics)
    ep_raw = raw_dir / canonical_id
    meta_files = list(ep_raw.glob("metadata_*.json")) if ep_raw.exists() else []
    if not meta_files:
        issues.append("raw metadata_*.json missing")
    else:
        meta = json.loads(meta_files[0].read_text())
        if "ext1_cam_extrinsics" not in meta:
            issues.append("ext1_cam_extrinsics missing from metadata")

    # 4. trajectory.h5 with wrist extrinsics key
    traj = ep_raw / "trajectory.h5"
    if not traj.exists():
        issues.append("trajectory.h5 missing")
    else:
        try:
            import h5py
            with h5py.File(traj, "r") as f:
                key = f"observation/camera_extrinsics/{wrist_serial}_left"
                if key not in f:
                    issues.append(f"wrist extrinsics key missing from trajectory.h5 ({key})")
        except Exception as e:
            issues.append(f"trajectory.h5 unreadable: {e}")

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-dir", required=True, type=Path)
    parser.add_argument("--raw-dir",   required=True, type=Path)
    parser.add_argument("--serial-map", type=Path,
                        help="Path to serial_map.json (default: {depth_dir}/serial_map.json)")
    parser.add_argument("--output-dir", type=Path,
                        help="Where to write results (default: {depth_dir})")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    serial_map_path = args.serial_map or (args.depth_dir / "serial_map.json")
    output_dir = args.output_dir or args.depth_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(serial_map_path) as f:
        serial_map = json.load(f)

    canonical_ids = sorted(serial_map.keys())
    print(f"Checking {len(canonical_ids)} episodes with {args.workers} workers...")

    valid = []
    invalid = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(check_episode, cid, args.depth_dir, args.raw_dir, serial_map): cid
            for cid in canonical_ids
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="checking"):
            cid = futures[fut]
            issues = fut.result()
            if issues:
                invalid[cid] = issues
            else:
                valid.append(cid)

    valid.sort()

    # Write results
    valid_path = output_dir / "valid_canonical_ids.json"
    invalid_path = output_dir / "invalid_episodes.json"
    with open(valid_path, "w") as f:
        json.dump(valid, f, indent=2)
    with open(invalid_path, "w") as f:
        json.dump(invalid, f, indent=2)

    # Summary
    print(f"\n{'='*50}")
    print(f"Valid:   {len(valid):4d} / {len(canonical_ids)} episodes")
    print(f"Invalid: {len(invalid):4d} / {len(canonical_ids)} episodes")
    print(f"\nWrote: {valid_path}")
    print(f"Wrote: {invalid_path}")

    if invalid:
        # Aggregate failure reasons
        from collections import Counter
        reason_counts = Counter()
        for issues in invalid.values():
            for issue in issues:
                # Normalise to category
                category = issue.split("(")[0].strip()
                reason_counts[category] += 1
        print("\nFailure reasons:")
        for reason, count in reason_counts.most_common():
            print(f"  {count:4d}x  {reason}")


if __name__ == "__main__":
    main()
