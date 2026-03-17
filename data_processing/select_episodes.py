"""
select_episodes.py

Selects the first N episodes (by episode index) from each of the specified labs
in the DROID dataset. Outputs a JSON file that can be consumed by other pipeline
scripts to filter processing to this subset.

Usage:
    python data_processing/select_episodes.py \
        --dataset-path /work/nvme/bgkz/droid_raw_large_superset \
        --output-file  /work/nvme/bgkz/droid_multilab_depths/selected_episodes.json \
        [--labs TRI AUTOLab IPRL REAL CLVR ILIAD IRIS] \
        [--max-per-lab 200]

Output JSON schema:
    {
        "labs": ["TRI", ...],
        "episode_indices": [0, 5, 12, ...],   // sorted ints
        "canonical_ids": ["TRI+...", ...],
        "per_lab_counts": {"TRI": 200, ...}
    }
"""

import argparse
import json
from pathlib import Path

DEFAULT_LABS = ["TRI", "AUTOLab", "IPRL", "REAL", "CLVR", "ILIAD", "IRIS"]
DEFAULT_MAX_PER_LAB = 200


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True, type=str,
                        help="Path to the LeRobot-format DROID dataset.")
    parser.add_argument("--output-file", required=True, type=str,
                        help="Path to write selected_episodes.json.")
    parser.add_argument("--labs", nargs="+", default=DEFAULT_LABS,
                        help="Lab names to include (matched as prefix of canonical_id).")
    parser.add_argument("--max-per-lab", type=int, default=DEFAULT_MAX_PER_LAB,
                        help="Maximum number of episodes to select per lab (sorted by episode index).")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"Dataset path not found: {dataset_path}")

    id_map_path = dataset_path / "meta" / "episode_index_to_id.json"
    if not id_map_path.exists():
        raise ValueError(f"episode_index_to_id.json not found at: {id_map_path}")

    with open(id_map_path) as f:
        id_map = json.load(f)

    # Bucket episode indices by lab
    lab_episodes: dict[str, list[tuple[int, str]]] = {lab: [] for lab in args.labs}
    all_labs_in_data: set[str] = set()

    for idx_str, meta in id_map.items():
        lab = meta["canonical_id"].split("+")[0]
        all_labs_in_data.add(lab)
        if lab in lab_episodes:
            lab_episodes[lab].append((int(idx_str), meta["canonical_id"]))

    # Validate: check all requested labs are present
    for lab in args.labs:
        if lab not in all_labs_in_data:
            raise ValueError(
                f"Lab '{lab}' not found in dataset metadata. "
                f"Available labs: {sorted(all_labs_in_data)}"
            )

    # Select first max_per_lab by ascending episode index
    selected_indices: list[int] = []
    selected_canonical: list[str] = []
    per_lab_counts: dict[str, int] = {}

    for lab in args.labs:
        episodes = sorted(lab_episodes[lab], key=lambda x: x[0])
        if len(episodes) < args.max_per_lab:
            print(
                f"[WARN] Lab '{lab}' has only {len(episodes)} episodes "
                f"(requested {args.max_per_lab}). Using all {len(episodes)}."
            )
        chosen = episodes[:args.max_per_lab]
        per_lab_counts[lab] = len(chosen)
        for idx, cid in chosen:
            selected_indices.append(idx)
            selected_canonical.append(cid)

    # Sort by episode index for deterministic ordering
    paired = sorted(zip(selected_indices, selected_canonical), key=lambda x: x[0])
    selected_indices = [p[0] for p in paired]
    selected_canonical = [p[1] for p in paired]

    output = {
        "labs": args.labs,
        "episode_indices": selected_indices,
        "canonical_ids": selected_canonical,
        "per_lab_counts": per_lab_counts,
    }

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    total = len(selected_indices)
    print(f"Selected {total} episodes across {len(args.labs)} labs:")
    for lab in args.labs:
        print(f"  {lab}: {per_lab_counts[lab]}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
