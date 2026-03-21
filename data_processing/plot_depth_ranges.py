"""
plot_depth_ranges.py

Plots per-lab distribution of depth values across all episodes in the depth cache.
One worker per episode; each loads ext1 + wrist depth.blosc, samples frames,
computes percentiles over valid pixels, and returns them keyed by lab.

Usage:
    python data_processing/plot_depth_ranges.py \
        --depth-dir /work/nvme/bgkz/droid_multilab_depths \
        --output depth_ranges.png \
        --workers 64 \
        --frames-per-episode 8
"""

import argparse
import json
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import blosc
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


PERCENTILES = [1, 5, 25, 50, 75, 95, 99]


def load_depth(depth_dir: Path, canonical_id: str, serial: str) -> np.ndarray | None:
    """Load (T, H, W) float32 from depth.blosc. Returns None on failure."""
    ep_path = depth_dir / canonical_id / serial
    try:
        shape = np.load(ep_path / "shape.npy")
        raw = (ep_path / "depth.blosc").read_bytes()
        return np.frombuffer(blosc.decompress(raw), dtype=np.float32).reshape(shape)
    except Exception:
        return None


def process_episode(args: tuple) -> tuple[str, np.ndarray] | None:
    """
    Worker: load ext1 + wrist depth for one episode, sample frames,
    compute percentiles over valid (finite, > 0) pixels.

    Returns (lab, percentiles_array [len(PERCENTILES)]) or None on failure.
    """
    canonical_id, depth_dir_str, serial_map_entry, frames_per_episode = args
    depth_dir = Path(depth_dir_str)
    lab = canonical_id.split("+")[0]

    all_valid_depths = []

    for camera in ("ext1", "wrist"):
        serial = serial_map_entry.get(camera)
        if serial is None:
            continue
        depth = load_depth(depth_dir, canonical_id, serial)
        if depth is None:
            continue

        T = depth.shape[0]
        frame_indices = np.linspace(0, T - 1, min(frames_per_episode, T), dtype=int)
        for fi in frame_indices:
            frame = depth[fi]
            valid = np.isfinite(frame) & (frame > 0)
            if valid.any():
                all_valid_depths.append(frame[valid])

    if not all_valid_depths:
        return None

    combined = np.concatenate(all_valid_depths)
    pcts = np.percentile(combined, PERCENTILES).astype(np.float32)
    return lab, pcts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="depth_ranges.png")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--frames-per-episode", type=int, default=8,
                        help="Frames to sample per episode per camera (for speed)")
    args = parser.parse_args()

    depth_dir = Path(args.depth_dir)

    serial_map = json.loads((depth_dir / "serial_map.json").read_text())

    # Only process episodes that have a serial_map entry (i.e. have ext1+wrist)
    all_canonical_ids = sorted(
        p.name for p in depth_dir.iterdir()
        if p.is_dir() and p.name in serial_map
    )
    print(f"Found {len(all_canonical_ids)} episodes with serial map entries")

    tasks = [
        (cid, str(depth_dir), serial_map[cid], args.frames_per_episode)
        for cid in all_canonical_ids
    ]

    # Collect per-lab percentile arrays
    lab_pcts: dict[str, list[np.ndarray]] = defaultdict(list)

    with Pool(processes=args.workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_episode, tasks, chunksize=4),
            total=len(tasks),
            desc="Episodes",
        ):
            if result is None:
                continue
            lab, pcts = result
            lab_pcts[lab].append(pcts)

    if not lab_pcts:
        raise ValueError("No valid depth data found — check --depth-dir")

    # Aggregate: median of each percentile across episodes per lab
    labs = sorted(lab_pcts.keys())
    print(f"\nLabs found: {labs}")

    # Summary table
    print(f"\n{'Lab':<15} {'N_eps':>6}  " + "  ".join(f"p{p:02d}" for p in PERCENTILES))
    for lab in labs:
        arr = np.stack(lab_pcts[lab])  # [N_eps, len(PERCENTILES)]
        medians = np.median(arr, axis=0)
        n = len(lab_pcts[lab])
        print(f"{lab:<15} {n:>6}  " + "  ".join(f"{v:5.2f}" for v in medians))

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Per-lab depth distribution (valid pixels, all frames sampled)", fontsize=13)

    # Panel 1: box plot of per-episode median depth
    ax = axes[0]
    medians_per_lab = [
        np.stack(lab_pcts[lab])[:, PERCENTILES.index(50)]
        for lab in labs
    ]
    ax.boxplot(medians_per_lab, labels=labs, vert=True, patch_artist=True)
    ax.set_ylabel("Depth (m)")
    ax.set_title("Median depth per episode")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(bottom=0)

    # Panel 2: band plot — median of p5/p50/p95 across episodes per lab
    ax2 = axes[1]
    x = np.arange(len(labs))
    p5_idx  = PERCENTILES.index(5)
    p50_idx = PERCENTILES.index(50)
    p95_idx = PERCENTILES.index(95)

    for i, lab in enumerate(labs):
        arr = np.stack(lab_pcts[lab])
        p5  = np.median(arr[:, p5_idx])
        p50 = np.median(arr[:, p50_idx])
        p95 = np.median(arr[:, p95_idx])
        ax2.bar(i, p95 - p5, bottom=p5, alpha=0.5, label=lab)
        ax2.scatter(i, p50, color="black", s=20, zorder=5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labs, rotation=45, ha="right")
    ax2.set_ylabel("Depth (m)")
    ax2.set_title("Depth range per lab  (bar=p5–p95, dot=p50)")
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
