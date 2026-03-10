"""
Check if episode_ids from the converted v2 dataset (episode_index_to_id.json)
are mappable to the droid_annotations (episode_id_to_path, language, camera_serials, etc.).

Our converter may store a full path (e.g. .../RAIL/success/2023-04-17/Mon_Apr_17_14:48:05_2023/trajectory.h5)
while annotations are keyed by canonical episode_id (e.g. RAIL+80edfcb1+2023-04-17-14h-48m-05s)
and episode_id_to_path has value = relative path (e.g. RAIL/success/2023-04-17/Mon_Apr_17_14:48:05_2023).
We derive the relative path from our stored id and look up the canonical episode_id.
"""
import json
import re
from pathlib import Path

ANNOTATIONS_DIR = Path("/ocean/projects/cis240058p/hbhatia1/data/droid_annotations")


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def our_path_to_relative(our_id: str) -> str | None:
    """
    Convert our stored episode id (often a full path to trajectory.h5) to the
    relative path form used as *value* in episode_id_to_path (e.g. site/outcome/date/folder).
    """
    s = our_id.strip().rstrip("/")
    if s.endswith("/trajectory.h5"):
        s = s[: -len("/trajectory.h5")]
    elif s.endswith("trajectory.h5"):
        s = s[: -len("trajectory.h5")].rstrip("/")
    # Take the last 4 path components: site/success|failure/date/folder
    parts = [p for p in s.split("/") if p]
    if len(parts) >= 4:
        return "/".join(parts[-4:])
    if len(parts) >= 1:
        return "/".join(parts)
    return None


def main():
    import argparse
    p = argparse.ArgumentParser(description="Check episode_id -> annotations mapping")
    p.add_argument(
        "--mapping",
        type=str,
        default="/ocean/projects/cis240058p/hbhatia1/data/droid_v2_small/meta/episode_index_to_id.json",
        help="Path to episode_index_to_id.json from converted dataset",
    )
    p.add_argument(
        "--annotations-dir",
        type=str,
        default=str(ANNOTATIONS_DIR),
    )
    args = p.parse_args()

    mapping_path = Path(args.mapping)
    annotations_dir = Path(args.annotations_dir)
    if not mapping_path.exists():
        print(f"Mapping file not found: {mapping_path}")
        return
    if not annotations_dir.exists():
        print(f"Annotations dir not found: {annotations_dir}")
        return

    our_mapping = load_json(mapping_path)
    # Our file may have string keys "0","1",...
    our_episode_ids = [our_mapping[str(k)] for k in sorted(int(x) for x in our_mapping)]

    episode_id_to_path = load_json(annotations_dir / "episode_id_to_path.json")
    # path (value) -> episode_id (key)
    path_to_episode_id = {v: k for k, v in episode_id_to_path.items()}

    lang_path = annotations_dir / "droid_language_annotations.json"
    camera_serials_path = annotations_dir / "camera_serials.json"
    language_annotations = load_json(lang_path) if lang_path.exists() else {}
    camera_serials = load_json(camera_serials_path) if camera_serials_path.exists() else {}

    n = len(our_episode_ids)
    n_path_match = 0
    n_language = 0
    n_camera_serials = 0
    canonical_ids = []

    for our_id in our_episode_ids:
        rel = our_path_to_relative(our_id)
        canonical = path_to_episode_id.get(rel) if rel else None
        if canonical is not None:
            n_path_match += 1
            canonical_ids.append(canonical)
            if canonical in language_annotations:
                n_language += 1
            if canonical in camera_serials:
                n_camera_serials += 1
        else:
            # Try direct match (in case we already store canonical id)
            if our_id in episode_id_to_path:
                canonical = our_id
                n_path_match += 1
                canonical_ids.append(canonical)
                if canonical in language_annotations:
                    n_language += 1
                if canonical in camera_serials:
                    n_camera_serials += 1

    print("Episode ID -> annotations mapping check")
    print("  Converted dataset mapping:", mapping_path)
    print("  Annotations dir:         ", annotations_dir)
    print("  Total episodes in v2:   ", n)
    print("  Match to episode_id_to_path (canonical id resolved):", n_path_match)
    print("  Of those, in droid_language_annotations:           ", n_language)
    print("  Of those, in camera_serials:                       ", n_camera_serials)
    if n > 0 and n_path_match == 0:
        print("\n  Sample our stored id:", our_episode_ids[0][:120] if len(our_episode_ids[0]) > 120 else our_episode_ids[0])
        print("  Derived relative path:", our_path_to_relative(our_episode_ids[0]))
        # Show a few annotation path values to compare
        sample_paths = list(path_to_episode_id.keys())[:3]
        print("  Sample annotation paths (values in episode_id_to_path):", sample_paths)
    if canonical_ids:
        print("\n  Sample canonical episode_id (first):", canonical_ids[0])


if __name__ == "__main__":
    main()
