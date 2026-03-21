"""Analyze camera extrinsic variation across labs in the DROID multi-lab dataset.

Reads ext1_cam_extrinsics and ext2_cam_extrinsics from metadata_*.json per episode,
and plots the per-lab distribution of (tx, ty, tz) for left vs right external camera.
"""

import argparse
import json
import pickle
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _process_episode(args: tuple) -> list[tuple[str, str, np.ndarray]] | None:
    """Return list of (lab, camera, xyz) tuples for one episode, or None on skip."""
    canonical_id, raw_dir_str = args
    raw_dir = Path(raw_dir_str)
    ep_dir = raw_dir / canonical_id

    meta_files = list(ep_dir.glob("metadata_*.json")) if ep_dir.exists() else []
    if not meta_files:
        warnings.warn(f"No metadata_*.json in {ep_dir} — skipping")
        return None

    try:
        meta = json.loads(meta_files[0].read_text())
    except Exception as e:
        warnings.warn(f"Failed to read {meta_files[0]}: {e} — skipping")
        return None

    lab = canonical_id.split("+")[0]
    results = []
    for cam_key, cam_name in (("ext1_cam_extrinsics", "ext1"), ("ext2_cam_extrinsics", "ext2")):
        dof = meta.get(cam_key)
        if dof is None:
            continue
        results.append((lab, cam_name, np.array(dof[:3], dtype=np.float64)))  # tx, ty, tz only

    return results if results else None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(extrinsics: dict, plot_path: str) -> None:
    """
    One figure, 3 subplots (tx, ty, tz).
    Each subplot: x-axis = labs, two box plots per lab (ext1=left, ext2=right).
    """
    labs = sorted(extrinsics.keys())
    axis_labels = ["tx (m)", "ty (m)", "tz (m)"]
    colors = {"ext1": "#4C72B0", "ext2": "#DD8452"}  # blue=left, orange=right

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for ax_idx, (ax, axis_label) in enumerate(zip(axes, axis_labels)):
        positions_ext1 = []
        positions_ext2 = []
        tick_positions = []
        tick_labels = []

        x = 0
        for lab in labs:
            data_ext1 = extrinsics[lab].get("ext1", [])
            data_ext2 = extrinsics[lab].get("ext2", [])

            vals_ext1 = [v[ax_idx] for v in data_ext1] if data_ext1 else []
            vals_ext2 = [v[ax_idx] for v in data_ext2] if data_ext2 else []

            if vals_ext1:
                bp = ax.boxplot(vals_ext1, positions=[x], widths=0.35,
                                patch_artist=True,
                                boxprops=dict(facecolor=colors["ext1"], alpha=0.7),
                                medianprops=dict(color="black", linewidth=1.5),
                                whiskerprops=dict(color=colors["ext1"]),
                                capprops=dict(color=colors["ext1"]),
                                flierprops=dict(marker=".", color=colors["ext1"],
                                                markersize=3, alpha=0.5))
            if vals_ext2:
                bp = ax.boxplot(vals_ext2, positions=[x + 0.4], widths=0.35,
                                patch_artist=True,
                                boxprops=dict(facecolor=colors["ext2"], alpha=0.7),
                                medianprops=dict(color="black", linewidth=1.5),
                                whiskerprops=dict(color=colors["ext2"]),
                                capprops=dict(color=colors["ext2"]),
                                flierprops=dict(marker=".", color=colors["ext2"],
                                                markersize=3, alpha=0.5))

            tick_positions.append(x + 0.2)
            tick_labels.append(lab)
            x += 1.2

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(axis_label)
        ax.set_title(axis_label)
        ax.grid(axis="y", alpha=0.3)

    # Shared legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=colors["ext1"], alpha=0.7, label="ext1 (left)"),
        Patch(facecolor=colors["ext2"], alpha=0.7, label="ext2 (right)"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=10)
    fig.suptitle("Camera position (tx, ty, tz) variation per lab", fontsize=13)
    fig.tight_layout()

    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DROID camera extrinsic variation")
    parser.add_argument("--raw-dir", type=Path,
                        default=Path("/work/nvme/bgkz/droid_multilab_raw"))
    parser.add_argument("--output", type=str, default="extrinsic_data.pkl")
    parser.add_argument("--plot", type=str, default="extrinsic_plots.png")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--load", action="store_true",
                        help="Skip collection and load from --output")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Collect (or load)
    # ------------------------------------------------------------------
    if args.load:
        print(f"Loading from {args.output} …")
        with open(args.output, "rb") as f:
            saved = pickle.load(f)
        extrinsics = saved["extrinsics"]
        print(f"  raw_dir={saved['raw_dir']}")
    else:
        raw_dir = args.raw_dir
        if not raw_dir.exists():
            raise ValueError(f"raw_dir does not exist: {raw_dir}")

        canonical_ids = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())
        if not canonical_ids:
            raise ValueError(f"No episode directories found in {raw_dir}")
        print(f"Found {len(canonical_ids)} episodes in {raw_dir}")

        tasks = [(cid, str(raw_dir)) for cid in canonical_ids]

        # {lab: {cam: [xyz, ...]}}
        extrinsics: dict[str, dict[str, list[np.ndarray]]] = {}

        with Pool(processes=args.workers) as pool:
            for result in tqdm(
                pool.imap_unordered(_process_episode, tasks, chunksize=8),
                total=len(tasks),
                desc="Collecting extrinsics",
            ):
                if result is None:
                    continue
                for lab, cam, xyz in result:
                    extrinsics.setdefault(lab, {}).setdefault(cam, []).append(xyz)

        if not extrinsics:
            raise ValueError("No extrinsic data collected — check --raw-dir")

        with open(args.output, "wb") as f:
            pickle.dump({"extrinsics": extrinsics, "raw_dir": str(raw_dir)}, f)
        print(f"Saved data to {args.output}")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print(f"\n{'Lab':<12} {'Camera':<8} {'N_episodes':>10}  {'tx mean':>8}  {'ty mean':>8}  {'tz mean':>8}")
    print("-" * 60)
    for lab in sorted(extrinsics.keys()):
        for cam in ("ext1", "ext2"):
            vals = extrinsics[lab].get(cam, [])
            if not vals:
                continue
            arr = np.stack(vals)
            print(f"{lab:<12} {cam:<8} {len(vals):>10}  "
                  f"{arr[:,0].mean():>8.3f}  {arr[:,1].mean():>8.3f}  {arr[:,2].mean():>8.3f}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    _plot(extrinsics, args.plot)


if __name__ == "__main__":
    main()
