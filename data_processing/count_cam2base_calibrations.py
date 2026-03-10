"""
Count how many cameras are calibrated in cam2base_extrinsics.json
and cam2base_extrinsic_superset.json.
"""
import json
from pathlib import Path
from collections import Counter

ANNOTATIONS_DIR = Path("/ocean/projects/cis240058p/hbhatia1/data/droid_annotations")


def is_camera_extrinsic_key(key: str, val) -> bool:
    """Key is a camera serial and value is 6-float extrinsics [tx,ty,tz,rx,ry,rz]."""
    if not isinstance(val, list) or len(val) != 6:
        return False
    if not all(isinstance(x, (int, float)) for x in val):
        return False
    # Exclude metadata keys (e.g. "24400334_metric_type")
    if "_" in key and key.split("_")[-1] in ("metric_type", "quality_metric", "source"):
        return False
    # Camera keys are typically numeric serials only (digits)
    return key.isdigit()


def count_calibrations(path: Path) -> dict:
    with path.open("r") as f:
        data = json.load(f)

    num_episodes = len(data)
    total_cameras = 0
    cameras_per_episode = []

    for episode_id, ep_data in data.items():
        if not isinstance(ep_data, dict):
            continue
        n = 0
        for k, v in ep_data.items():
            if is_camera_extrinsic_key(k, v):
                n += 1
        total_cameras += n
        cameras_per_episode.append(n)

    hist = Counter(cameras_per_episode)
    return {
        "file": path.name,
        "num_episodes": num_episodes,
        "total_camera_calibrations": total_cameras,
        "cameras_per_episode": dict(sorted(hist.items())),
    }


def main():
    files = [
        ANNOTATIONS_DIR / "cam2base_extrinsics.json",
        ANNOTATIONS_DIR / "cam2base_extrinsic_superset.json",
    ]
    for path in files:
        if not path.exists():
            print(f"Skip (not found): {path}")
            continue
        print(f"Loading {path.name} ...")
        stats = count_calibrations(path)
        print(f"  {path.name}")
        print(f"    Episodes:                    {stats['num_episodes']:,}")
        print(f"    Total camera calibrations:   {stats['total_camera_calibrations']:,}")
        print(f"    Cameras per episode (hist):  {stats['cameras_per_episode']}")
        print()


if __name__ == "__main__":
    main()
