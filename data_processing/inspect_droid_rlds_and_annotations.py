import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def print_annotation_summary(annotations_dir: Path) -> None:
    print(f"Using annotations_dir={annotations_dir}")
    if not annotations_dir.exists():
        print("WARNING: annotations_dir does not exist on disk.")
        return

    epi_map_path = annotations_dir / "episode_id_to_path.json"
    keep_ranges_path = annotations_dir / "keep_ranges_1_0_1.json"
    lang_path = annotations_dir / "droid_language_annotations.json"

    print("\n=== Annotation files present ===")
    for name in [
        "intrinsics.json",
        "cam2cam_extrinsics.json",
        "cam2base_extrinsics.json",
        "cam2base_extrinsic_superset.json",
        "camera_serials.json",
        "episode_id_to_path.json",
        "keep_ranges_1_0_1.json",
        "droid_language_annotations.json",
    ]:
        p = annotations_dir / name
        print(f"{name}: {'OK' if p.exists() else 'MISSING'}")

    epi_map: Optional[Dict[str, Any]] = None
    keep_ranges: Optional[Dict[str, Any]] = None
    lang_ann: Optional[Dict[str, Any]] = None

    if epi_map_path.exists():
        epi_map = load_json(epi_map_path)
        print(f"\nLoaded episode_id_to_path.json with {len(epi_map)} entries.")
        for i, (k, v) in enumerate(epi_map.items()):
            print(f"  example episode_id_to_path[{k!r}] -> {v!r}")
            if i >= 4:
                break

    if keep_ranges_path.exists():
        keep_ranges = load_json(keep_ranges_path)
        print(f"\nLoaded keep_ranges_1_0_1.json with {len(keep_ranges)} episode entries.")
        for i, (k, ranges) in enumerate(keep_ranges.items()):
            print(f"  episode {k!r} has {len(ranges)} keep ranges")
            if i >= 4:
                break

    if lang_path.exists():
        lang_ann = load_json(lang_path)
        print(f"\nLoaded droid_language_annotations.json with {len(lang_ann)} entries.")
        for i, (k, v) in enumerate(lang_ann.items()):
            print(f"  language entry for {k!r}: {str(v)[:120]!r}...")
            if i >= 2:
                break


def inspect_rlds(source_dir: Path, max_episodes: int = 3) -> None:
    """Attempt to inspect a few RLDS episodes using tensorflow_datasets, if available.

    This is best-effort: if tensorflow_datasets or the DROID builder are not
    available, the function will print a warning and return without failing
    the whole script.
    """
    print("\n=== RLDS inspection ===")
    print(f"Using source_dir={source_dir}")

    if not source_dir.exists():
        print("WARNING: source_dir does not exist on disk. Skipping RLDS inspection.")
        return

    try:
        import tensorflow_datasets as tfds  # type: ignore
    except ImportError:
        print("tensorflow_datasets is not installed in this environment. Skipping RLDS episode inspection.")
        return

    # Try to guess the dataset name; DROID docs expose 'droid' and 'droid_100'.
    possible_names = ["droid_100", "droid"]
    ds = None
    used_name = None

    for name in possible_names:
        try:
            print(f"Trying to load dataset '{name}' from data_dir={source_dir}...")
            ds = tfds.load(
                name,
                data_dir=str(source_dir),
                split="train",
                read_config=tfds.ReadConfig(try_autocache=False),
            )
            used_name = name
            break
        except Exception as e:  # noqa: BLE001
            print(f"  Failed to load '{name}': {e}")

    if ds is None:
        print("Could not load a DROID RLDS dataset from source_dir using tensorflow_datasets.")
        return

    print(f"Successfully loaded dataset '{used_name}' from {source_dir}.")
    print(f"Inspecting up to {max_episodes} episodes...\n")

    # Episodes are sequences; print basic info and keys for a few of them.
    for epi_idx, episode in enumerate(tfds.as_numpy(ds.take(max_episodes))):
        print(f"--- Episode {epi_idx} ---")
        if isinstance(episode, dict) and "steps" in episode:
            steps = episode["steps"]
            num_steps = steps["is_first"].shape[0]
            print(f"  num_steps: {num_steps}")
            obs = steps.get("observation", {})
            if isinstance(obs, dict):
                cam_keys = [k for k in obs.keys() if "image" in k]
                print(f"  cameras: {cam_keys}")
                for cam in cam_keys[:3]:
                    shape = obs[cam].shape
                    print(f"    {cam}: shape={shape}")
            actions = steps.get("action", None)
            if actions is not None:
                print(f"  action shape: {actions.shape}")

            lang = steps.get("language_instruction", None)
            if lang is not None:
                # language_instruction is typically per-step; show first non-empty example
                try:
                    first_non_empty = next(
                        (s.decode("utf-8") for s in lang if getattr(s, "__len__", lambda: 0)() > 0),
                        "",
                    )
                except Exception:  # noqa: BLE001
                    first_non_empty = ""
                if first_non_empty:
                    print(f"  example language_instruction: {first_non_empty!r}")
        else:
            print(
                "  Unexpected episode structure (no 'steps' key). Keys:",
                list(episode.keys()) if isinstance(episode, dict) else type(episode),
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect DROID RLDS data and droid_annotations for compatibility."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="/ocean/projects/cis240058p/hbhatia1/data/droid_raw_small",
        help="Root directory containing the local RLDS copy (e.g., droid_100). Adjust to where you stored the gsutil copy.",
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="/ocean/projects/cis240058p/hbhatia1/data/droid_annotations",
        help="Directory containing droid_annotations JSON files.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=3,
        help="Maximum number of episodes to inspect from RLDS dataset.",
    )
    args = parser.parse_args()

    source_dir = Path(os.path.expanduser(args.source_dir))
    annotations_dir = Path(os.path.expanduser(args.annotations_dir))

    print_annotation_summary(annotations_dir)
    inspect_rlds(source_dir, max_episodes=args.max_episodes)


if __name__ == "__main__":
    main()

