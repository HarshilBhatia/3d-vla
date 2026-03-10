"""
Inspect cam2cam_extrinsics.json: structure, which cameras are present per episode,
and summary counts. Confirms whether wrist camera appears in cam2cam.
"""
import json
from pathlib import Path
from collections import Counter

ANNOTATIONS_DIR = Path("/ocean/projects/cis240058p/hbhatia1/data/droid_annotations")
CAM2CAM_PATH = ANNOTATIONS_DIR / "cam2cam_extrinsics.json"


def is_camera_block(key: str, val) -> bool:
    """True if key looks like a camera entry (e.g. left_cam, right_cam) and val has pose."""
    if key in ("relative_path", "metric_type", "quality_metric"):
        return False
    if not isinstance(val, dict):
        return False
    return "pose" in val and "focal" in val


def main():
    if not CAM2CAM_PATH.exists():
        print(f"Not found: {CAM2CAM_PATH}")
        return

    print(f"Loading {CAM2CAM_PATH.name} (may take a moment)...")
    with CAM2CAM_PATH.open("r") as f:
        data = json.load(f)

    num_episodes = len(data)
    cameras_per_episode = []
    camera_keys_seen = Counter()
    pose_shapes = []

    for episode_id, ep_data in data.items():
        if not isinstance(ep_data, dict):
            continue
        cam_keys = [k for k, v in ep_data.items() if is_camera_block(k, v)]
        for k in cam_keys:
            camera_keys_seen[k] += 1
        n = len(cam_keys)
        cameras_per_episode.append(n)
        # Sample pose shape from first episode's first camera
        if pose_shapes == [] and cam_keys:
            pose = ep_data[cam_keys[0]].get("pose")
            if isinstance(pose, list):
                pose_shapes.append((len(pose), len(pose[0]) if pose else 0))

    hist = Counter(cameras_per_episode)
    print()
    print("cam2cam_extrinsics.json")
    print("  Episodes:                      ", f"{num_episodes:,}")
    print("  Cameras per episode (hist):    ", dict(sorted(hist.items())))
    print("  Camera keys (name -> count):   ", dict(camera_keys_seen))
    if pose_shapes:
        print("  Pose shape (sample 4x4):        ", pose_shapes[0])
    print()

    # Explicit wrist check
    if "wrist_cam" in camera_keys_seen or "wrist" in str(camera_keys_seen).lower():
        print("  Wrist camera: present in cam2cam.")
    else:
        print("  Wrist camera: NOT present in cam2cam (only left_cam / right_cam = exterior pair).")
    print()

    # Sample one episode's structure
    first_ep = next(iter(data.values()))
    print("  Sample episode keys (top-level):", list(first_ep.keys()))
    cam_key = next((k for k in first_ep if is_camera_block(k, first_ep[k])), None)
    if cam_key:
        print(f"  Sample camera block '{cam_key}' keys:", list(first_ep[cam_key].keys()))


if __name__ == "__main__":
    main()
