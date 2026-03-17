"""
download_multilab_depths.py

Downloads SVO (depth), trajectory.h5 (extrinsics), and metadata JSON for a
set of episodes specified in a selected_episodes.json file (produced by
select_episodes.py). Generalised version of download_rail_depths.py.

Per episode, downloads to {output_dir}/{canonical_id}/:
  recordings/SVO/*.svo   -- ZED stereo recordings (depth source)
  trajectory.h5          -- per-timestep extrinsics + robot state
  metadata_*.json        -- serial -> camera mapping + static extrinsics

Usage:
    python data_processing/download_multilab_depths.py \
        --selected-episodes-file /work/nvme/bgkz/droid_multilab_depths/selected_episodes.json \
        --output-dir /work/nvme/bgkz/droid_multilab_raw \
        [--num-workers 32] \
        [--dry-run]
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


GCS_BASE    = "gs://gresearch/robotics/droid_raw/1.0.1"
XEMB_PREFIX = "gs://xembodiment_data/r2d2/r2d2-data-full/"
META_PATH   = "/work/nvme/bgkz/droid_raw_large_superset/meta/episode_index_to_id.json"


def get_episodes_from_selected(selected_file: Path, meta_path: str) -> list[dict]:
    """Build a list of episode dicts from selected_episodes.json + episode_index_to_id.json."""
    if not selected_file.exists():
        raise ValueError(f"Selected-episodes file not found: {selected_file}")

    with open(selected_file) as f:
        selected = json.load(f)

    selected_indices = set(selected["episode_indices"])

    with open(meta_path) as f:
        id_map = json.load(f)

    episodes = []
    for idx_str, meta in id_map.items():
        if int(idx_str) not in selected_indices:
            continue
        rel = meta["stored_id"].replace(XEMB_PREFIX, "").replace("/trajectory.h5", "")
        gcs_path = f"{GCS_BASE}/{rel}"
        episodes.append({
            "ep_idx": int(idx_str),
            "canonical_id": meta["canonical_id"],
            "gcs_path": gcs_path,
        })

    return episodes


def gsutil_cp(src: str, dst_dir: Path, recursive: bool = False) -> tuple[bool, str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["gsutil", "-m", "cp"]
    if recursive:
        cmd.append("-r")
    cmd += [src, str(dst_dir) + "/"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr.strip()


def download_episode(ep: dict, output_dir: Path) -> tuple[str, bool, str]:
    ep_id = ep["canonical_id"]
    gcs   = ep["gcs_path"]
    local = output_dir / ep_id
    local.mkdir(parents=True, exist_ok=True)

    done_file = local / ".done"
    if done_file.exists():
        return ep_id, True, "already done"

    errors = []

    # 1. SVO files (contain depth) — recursive dir copy
    ok, err = gsutil_cp(f"{gcs}/recordings/SVO/*", local / "recordings/SVO")
    if not ok:
        errors.append(f"SVO: {err}")

    # 2. trajectory.h5 (per-timestep extrinsics)
    ok, err = gsutil_cp(f"{gcs}/trajectory.h5", local)
    if not ok:
        errors.append(f"trajectory.h5: {err}")

    # 3. metadata JSON (serial -> camera mapping)
    ok, err = gsutil_cp(f"{gcs}/metadata_*.json", local)
    if not ok:
        errors.append(f"metadata: {err}")

    if errors:
        return ep_id, False, " | ".join(errors)

    done_file.touch()
    return ep_id, True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-episodes-file", required=True, type=Path,
                        help="Path to selected_episodes.json from select_episodes.py")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Root directory for raw episode downloads")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--meta-path", default=META_PATH,
                        help="Path to episode_index_to_id.json")
    args = parser.parse_args()

    if not args.selected_episodes_file.exists():
        raise ValueError(f"Selected-episodes file not found: {args.selected_episodes_file}")

    episodes = get_episodes_from_selected(args.selected_episodes_file, args.meta_path)
    print(f"Found {len(episodes)} episodes to download")

    if args.dry_run:
        for ep in episodes[:3]:
            g = ep["gcs_path"]
            l = args.output_dir / ep["canonical_id"]
            print(f"  {g}/recordings/SVO/ -> {l}/recordings/SVO/")
            print(f"  {g}/trajectory.h5   -> {l}/trajectory.h5")
            print(f"  {g}/metadata_*.json -> {l}/")
        if len(episodes) > 3:
            print(f"  ... and {len(episodes) - 3} more")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    done, failed = 0, 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(download_episode, ep, args.output_dir): ep for ep in episodes}
        for fut in as_completed(futures):
            ep_id, ok, msg = fut.result()
            if ok:
                done += 1
                if done % 50 == 0 or msg != "ok":
                    print(f"[{done}/{len(episodes)}] {ep_id}: {msg}")
            else:
                failed += 1
                print(f"FAILED {ep_id}: {msg}", file=sys.stderr)

    print(f"\nDone: {done} ok, {failed} failed")


if __name__ == "__main__":
    main()
