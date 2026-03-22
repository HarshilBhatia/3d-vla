"""
Convert RLBench dataset_generator output to Peract_packaged layout (.dat files in train/val, task+var/).

Use after collecting with --camera_rig_rotation_deg 10. Then run peract_to_zarr on the output with no transform.

Usage:
  python scripts/rlbench/rlbench_to_peract_packaged.py \
    --rlbench_save_path peract_raw_rot10 \
    --out peract_raw_rot10/Peract_packaged
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import blosc

# RLBench layout
EPISODES_FOLDER = "episodes"
EPISODE_FOLDER = "episode%d"
LOW_DIM_PICKLE = "low_dim_obs.pkl"
VARIATION_NUMBER = "variation_number.pkl"
IMAGE_FORMAT = "%d.png"
DEPTH_SCALE = 2**24 - 1
PERACT_CAMERAS = ["left_shoulder", "right_shoulder", "wrist", "front"]
IM_SIZE = 256

PERACT_TASKS = [
    "place_cups", "close_jar", "insert_onto_square_peg",
    "light_bulb_in", "meat_off_grill", "open_drawer",
    "place_shape_in_shape_sorter", "place_wine_at_rack_location",
    "push_buttons", "put_groceries_in_cupboard",
    "put_item_in_drawer", "put_money_in_safe", "reach_and_drag",
    "slide_block_to_color_target", "stack_blocks", "stack_cups",
    "sweep_to_dustpan_of_size", "turn_tap",
]


def image_to_float_array(image_array, scale_factor):
    """Decode depth from RGB image (same as rlbench.backend.utils)."""
    image_array = np.array(image_array)
    if len(image_array.shape) == 3:
        float_array = np.sum(image_array * [65536, 256, 1], axis=2)
    else:
        float_array = image_array
    return float_array.astype(np.float64) / scale_factor


def depth_to_pointcloud(depth_m, intrinsics, extrinsics):
    """Convert depth (H,W) in meters to point cloud (H,W,3) in world frame."""
    H, W = depth_m.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u = np.arange(W, dtype=np.float64)
    v = np.arange(H, dtype=np.float64)
    u, v = np.meshgrid(u, v)
    x_c = (u - cx) * depth_m / fx
    y_c = (v - cy) * depth_m / fy
    z_c = depth_m
    p_c = np.stack([x_c, y_c, z_c], axis=-1)
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    p_w = (R @ p_c.reshape(-1, 3).T).T + t
    return p_w.reshape(H, W, 3).astype(np.float32)


def get_keyframe_indices(demo):
    """Return indices of keyframe steps: gripper state changes + last frame.

    Matches standard PerAct/3DDA preprocessing: only decision-point observations
    are stored (~5-10 per episode), keeping .dat files small (~5-15 MB vs 600+ MB
    for full trajectories).
    """
    keyframes = []
    prev_open = demo[0].gripper_open
    for i in range(1, len(demo)):
        if demo[i].gripper_open != prev_open:
            keyframes.append(i)
        prev_open = demo[i].gripper_open
    last = len(demo) - 1
    if not keyframes or keyframes[-1] != last:
        keyframes.append(last)
    return keyframes


def load_one_frame(rlbench_episode_path, frame_idx, demo_obs, image_size, rlbench_depth_decode):
    """Load RGB + PCD for all PERACT_CAMERAS at a single frame index."""
    from PIL import Image
    rgb_cams = []
    pcd_cams = []
    for cam in PERACT_CAMERAS:
        rgb_path = os.path.join(rlbench_episode_path, f"{cam}_rgb", IMAGE_FORMAT % frame_idx)
        depth_path = os.path.join(rlbench_episode_path, f"{cam}_depth", IMAGE_FORMAT % frame_idx)
        if not os.path.isfile(rgb_path) or not os.path.isfile(depth_path):
            raise FileNotFoundError(f"Missing {rgb_path} or {depth_path}")
        rgb_img = np.array(Image.open(rgb_path))
        if rgb_img.ndim == 2:
            rgb_img = np.stack([rgb_img] * 3, axis=-1)
        depth_img = Image.open(depth_path)
        depth_01 = rlbench_depth_decode(depth_img, DEPTH_SCALE)
        near = demo_obs.misc.get(f"{cam}_camera_near", 0.1)
        far = demo_obs.misc.get(f"{cam}_camera_far", 2.0)
        depth_m = near + depth_01 * (far - near)
        extrinsics = demo_obs.misc[f"{cam}_camera_extrinsics"]
        intrinsics = demo_obs.misc[f"{cam}_camera_intrinsics"]
        if depth_m.shape[0] != image_size[0] or depth_m.shape[1] != image_size[1]:
            from PIL import Image as PImage
            depth_m = np.array(
                PImage.fromarray((depth_m * 1000).astype(np.uint16)).resize(
                    (image_size[1], image_size[0]), resample=PImage.NEAREST
                )
            ) / 1000.0
        pc = depth_to_pointcloud(depth_m, intrinsics, extrinsics)
        if rgb_img.shape[0] != image_size[0] or rgb_img.shape[1] != image_size[1]:
            rgb_img = np.array(
                Image.fromarray(rgb_img).resize((image_size[1], image_size[0]), Image.BILINEAR)
            )
        rgb_cams.append(rgb_img)
        pcd_cams.append(pc)
    return rgb_cams, pcd_cams


def load_episode(rlbench_episode_path, image_size=(256, 256)):
    """Load one RLBench episode: extract keyframes only, return content list for .dat.

    Stores only keyframe observations (~5-10 per episode) matching the original
    Peract_packaged format from HuggingFace, keeping file sizes ~5-15 MB each.
    """
    with open(os.path.join(rlbench_episode_path, LOW_DIM_PICKLE), "rb") as f:
        demo = pickle.load(f)
    with open(os.path.join(rlbench_episode_path, VARIATION_NUMBER), "rb") as f:
        var = pickle.load(f)

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        raise ImportError("PIL is required; pip install Pillow")

    try:
        from rlbench.backend.utils import image_to_float_array as rlbench_depth_decode
    except ImportError:
        rlbench_depth_decode = lambda img, scale: image_to_float_array(np.array(img), scale)

    keyframes = get_keyframe_indices(demo)
    T = len(keyframes)

    rgb_frames = []
    pcd_frames = []
    for k in keyframes:
        rgb_cams, pcd_cams = load_one_frame(
            rlbench_episode_path, k, demo[k], image_size, rlbench_depth_decode
        )
        # stack cameras: (NCAM, H, W, 3) → (NCAM, 3, H, W)
        rgb_frames.append(np.stack(rgb_cams, axis=0).transpose(0, 3, 1, 2))
        pcd_frames.append(np.stack(pcd_cams, axis=0).transpose(0, 3, 1, 2))

    # (T, NCAM, 3, H, W)
    rgb = np.stack(rgb_frames, axis=0).astype(np.float32)
    pcd = np.stack(pcd_frames, axis=0).astype(np.float32)
    # Normalise RGB to [-1, 1]
    rgb_norm = (rgb / 127.5) - 1.0
    # content[1]: (T, NCAM, 2, 3, H, W)
    content1 = np.concatenate([rgb_norm[:, :, np.newaxis], pcd[:, :, np.newaxis]], axis=2)

    # content[0]: original trajectory indices of the keyframes
    content0 = keyframes

    # content[2]: action at keyframe i = NEXT keyframe's gripper pose+open
    actions = []
    for idx in range(T):
        next_k = keyframes[idx + 1] if idx + 1 < T else keyframes[-1]
        o = demo[next_k]
        pose = np.array(o.gripper_pose, dtype=np.float32)
        open_ = np.array([float(o.gripper_open)], dtype=np.float32)
        actions.append(np.concatenate([pose, open_]).reshape(1, 8))
    content2 = actions

    # content[4]: current gripper pose+open at each keyframe
    prop = []
    for k in keyframes:
        o = demo[k]
        pose = np.array(o.gripper_pose, dtype=np.float32)
        open_ = np.array([float(o.gripper_open)], dtype=np.float32)
        prop.append(np.concatenate([pose, open_]).reshape(1, 8))
    content4 = prop

    content_list = [content0, content1, content2, None, content4]
    return content_list, var


def main():
    parser = argparse.ArgumentParser(description="Convert RLBench output to Peract_packaged .dat layout")
    parser.add_argument("--rlbench_save_path", type=str, required=True, help="Path from dataset_generator --save_path")
    parser.add_argument("--out", type=str, required=True, help="Output root (will create train/ and val/ with task+var/*.dat)")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256], help="H W for resizing")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Fraction of episodes to put in val (by variation)")
    parser.add_argument("--ep_offset", type=int, default=0,
                        help="Add this offset to episode indices when naming output .dat files. "
                             "Use when merging multiple batches: batch 0 → offset 0, batch 1 → offset 25, etc.")
    args = parser.parse_args()

    root = Path(args.rlbench_save_path)
    out = Path(args.out)
    if not root.is_dir():
        print(f"[ERROR] Not a directory: {root}")
        sys.exit(1)
    image_size = tuple(args.image_size)
    val_ratio = args.val_ratio
    ep_offset = args.ep_offset
    if ep_offset:
        print(f"[INFO] Episode offset: {ep_offset} (output files will be ep{ep_offset}.dat, ep{ep_offset+1}.dat, ...)")

    # Discover tasks (must match PERACT_TASKS for peract_to_zarr)
    all_variations = "all_variations"
    episodes_folder = EPISODES_FOLDER
    task_dirs = [d for d in root.iterdir() if d.is_dir()]
    converted = 0
    for task_dir in sorted(task_dirs):
        task_name = task_dir.name
        if task_name not in PERACT_TASKS:
            continue
        var_ep_path = task_dir / all_variations / episodes_folder
        if not var_ep_path.is_dir():
            continue
        for ep_dir in sorted(var_ep_path.iterdir()):
            # RLBench writes episode0, episode1, ... (no underscore)
            if not ep_dir.is_dir() or not ep_dir.name.startswith("episode"):
                continue
            pkl = ep_dir / LOW_DIM_PICKLE
            var_pkl = ep_dir / VARIATION_NUMBER
            if not pkl.is_file() or not var_pkl.is_file():
                continue
            try:
                content_list, var = load_episode(str(ep_dir), image_size=image_size)
            except Exception as e:
                print(f"[WARN] Skip {ep_dir}: {e}")
                continue
            # Train/val split by variation
            split = "val" if (var % 5 == 0 and val_ratio > 0) else "train"
            folder = out / split / f"{task_name}+{var}"
            folder.mkdir(parents=True, exist_ok=True)
            ep_idx = int(ep_dir.name.replace("episode", "")) + ep_offset
            dat_path = folder / f"ep{ep_idx}.dat"
            with open(dat_path, "wb") as f:
                f.write(blosc.compress(pickle.dumps(content_list)))
            converted += 1
    print(f"[OK] Converted {converted} episodes to {out}/ (train/ and val/)")
    if converted == 0:
        print("[WARN] No episodes found. Check --rlbench_save_path and that dataset_generator was run with PerAct tasks.")


if __name__ == "__main__":
    main()
