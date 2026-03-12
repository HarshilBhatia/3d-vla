"""
cache_3d_positions.py

Pre-computes 3D world-frame positions for each image token in the backbone feature cache.
Run this AFTER cache_backbone_features.py (same shard layout required).

For each cached sample, this script:
  1. Reads image_grid_thw to determine patch layout per camera.
  2. For the exterior camera: loads the pre-extracted depth frame (.npy), scales intrinsics,
     and unprojects each 14×14 patch region into world-frame (x, y, z) using extrinsics.
  3. For wrist camera tokens and text tokens: assigns position [0, 0, 0] (identity RoPE).
  4. Pads all positions to the shard's max seq_len (same padding as backbone shard).

Output layout (one file per shard, matching backbone shard layout):
    {pos_cache_dir}/
        pos_shard_{idx:04d}.pt   # {"token_positions_3d": Tensor[N, seq_len, 3]} float32
        pos_shard_{idx:04d}.done # sentinel

Depth frames are pre-extracted .npy files (float32, shape [H, W]) at:
    {depth_dir}/{episode_id}/depth_{frame_idx:06d}.npy

Intrinsics JSON: { "episode_id": {"fx": ..., "fy": ..., "cx": ..., "cy": ...} }
  - Based on original 180×320 (H×W) image resolution.

Extrinsics JSON: { "episode_id": { "frame_idx": [[4×4 T_cam2base]] } }
  - Per-timestep 4×4 homogeneous transform from camera frame to robot base frame.
  - Alternatively uses trajectory.h5 (future work for wrist camera).

Usage (SLURM array job, 37 tasks for RAIL):
    sbatch --array=0-36 scripts/slurm/cache_3d_positions.slurm
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS


PATCH_SIZE = 14          # Eagle uses 14×14 pixel patches
ORIG_H, ORIG_W = 180, 320  # Original DROID image resolution


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True,
                        help="Backbone feature cache dir (reads index.pt for shard layout)")
    parser.add_argument("--pos-cache-dir", type=str, required=True,
                        help="Output directory for 3D position shards")
    parser.add_argument("--depth-dir", type=str, required=True,
                        help="Root dir for pre-extracted depth .npy files: "
                             "{depth_dir}/{episode_id}/depth_{frame_idx:06d}.npy")
    parser.add_argument("--intrinsics-json", type=str, required=True,
                        help="JSON: {episode_id: {fx, fy, cx, cy}} at orig 180×320 resolution")
    parser.add_argument("--extrinsics-json", type=str, required=True,
                        help="JSON: {episode_id: {frame_idx: [[4×4 T_cam2base]]}} "
                             "(or trajectory.h5 format in future)")
    parser.add_argument("--embodiment-tag", type=str, default="OXE_DROID")
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--episode-sampling-rate", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--labs", type=str, nargs="+", default=None)
    return parser.parse_args()


def get_allowed_episode_indices(dataset_path: str, labs: list) -> set:
    id_map_path = Path(dataset_path) / "meta" / "episode_index_to_id.json"
    id_map = json.load(open(id_map_path))
    allowed = set()
    lab_set = set(labs)
    for idx_str, meta in id_map.items():
        lab = meta["canonical_id"].split("+")[0]
        if lab in lab_set:
            allowed.add(int(idx_str))
    return allowed


def unproject_patch(
    depth_resized: np.ndarray,  # [resized_H, resized_W] float32 meters
    patch_i: int,
    patch_j: int,
    fx_new: float,
    fy_new: float,
    cx_new: float,
    cy_new: float,
    T_cam2base: np.ndarray,  # [4, 4]
) -> np.ndarray:
    """Unproject a 14×14 patch to a 3D world-frame point (average over valid pixels).

    Returns [x, y, z] in the robot base frame, or [0, 0, 0] if all depths are invalid.
    """
    r0 = patch_i * PATCH_SIZE
    c0 = patch_j * PATCH_SIZE
    patch_depth = depth_resized[r0 : r0 + PATCH_SIZE, c0 : c0 + PATCH_SIZE]  # [14, 14]

    valid = patch_depth > 0
    if not valid.any():
        return np.zeros(3, dtype=np.float32)

    rows, cols = np.where(valid)
    u = cols.astype(np.float32) + c0  # pixel x (column) in resized image
    v = rows.astype(np.float32) + r0  # pixel y (row) in resized image
    d = patch_depth[rows, cols]

    X_cam = (u - cx_new) * d / fx_new
    Y_cam = (v - cy_new) * d / fy_new
    Z_cam = d

    # Average over valid pixels → single point per patch
    pts_cam = np.stack([X_cam, Y_cam, Z_cam, np.ones_like(d)], axis=1)  # [N, 4]
    pt_cam_mean = pts_cam.mean(axis=0)  # [4]

    pt_world = T_cam2base @ pt_cam_mean  # [4]
    return pt_world[:3].astype(np.float32)


def compute_positions_for_sample(
    image_grid_thw: np.ndarray,  # [num_images, 3] (T, H_patches, W_patches)
    image_mask: np.ndarray,       # [seq_len] bool
    depth_path: Optional[Path],   # path to depth .npy, or None
    intrinsics: Optional[dict],   # {fx, fy, cx, cy} at orig 180×320
    T_cam2base: Optional[np.ndarray],  # [4, 4] or None
) -> np.ndarray:
    """Build token_positions_3d [seq_len, 3] for one sample.

    Exterior camera (index 0): unproject patches using depth + intrinsics + extrinsics.
    Wrist camera (index 1) and text tokens: position [0, 0, 0].
    """
    seq_len = len(image_mask)
    positions = np.zeros((seq_len, 3), dtype=np.float32)

    # Count tokens per image for ordering
    # image_grid_thw[i] = [T, H_patches, W_patches]
    num_tokens_per_image = [
        int(image_grid_thw[i, 0]) * int(image_grid_thw[i, 1]) * int(image_grid_thw[i, 2])
        for i in range(len(image_grid_thw))
    ]

    # Locate image token positions in the sequence
    # Image tokens are contiguous blocks in the order they appear in the conversation.
    img_token_indices = np.where(image_mask)[0]
    if len(img_token_indices) == 0:
        return positions

    # Check if we have exterior camera depth available
    has_depth = (
        depth_path is not None
        and depth_path.exists()
        and intrinsics is not None
        and T_cam2base is not None
    )

    if has_depth and len(image_grid_thw) >= 1:
        # Exterior camera = first image in the conversation
        n_ext = num_tokens_per_image[0]
        ext_token_indices = img_token_indices[:n_ext]

        # Load and resize depth map
        depth = np.load(depth_path).astype(np.float32)  # [H, W]

        thw = image_grid_thw[0]
        patch_h, patch_w = int(thw[1]), int(thw[2])
        resized_H = patch_h * PATCH_SIZE
        resized_W = patch_w * PATCH_SIZE

        if depth.shape != (resized_H, resized_W):
            depth_resized = cv2.resize(depth, (resized_W, resized_H), interpolation=cv2.INTER_LINEAR)
        else:
            depth_resized = depth

        # Scale intrinsics from original 180×320 to resized resolution
        fx_new = intrinsics["fx"] * resized_W / ORIG_W
        fy_new = intrinsics["fy"] * resized_H / ORIG_H
        cx_new = intrinsics["cx"] * resized_W / ORIG_W
        cy_new = intrinsics["cy"] * resized_H / ORIG_H

        T = np.array(T_cam2base, dtype=np.float32)

        for k, seq_idx in enumerate(ext_token_indices):
            patch_i = k // patch_w
            patch_j = k % patch_w
            positions[seq_idx] = unproject_patch(
                depth_resized, patch_i, patch_j, fx_new, fy_new, cx_new, cy_new, T
            )

    # Wrist camera and text tokens stay at [0, 0, 0] (identity RoPE)
    return positions


def main():
    args = parse_args()
    pos_cache_dir = Path(args.pos_cache_dir)
    pos_cache_dir.mkdir(parents=True, exist_ok=True)

    # Load lookup tables
    print(f"[rank {args.rank}] Loading intrinsics from {args.intrinsics_json}")
    with open(args.intrinsics_json) as f:
        all_intrinsics = json.load(f)

    print(f"[rank {args.rank}] Loading extrinsics from {args.extrinsics_json}")
    with open(args.extrinsics_json) as f:
        all_extrinsics = json.load(f)

    # Load episode_id map
    id_map_path = Path(args.dataset_path) / "meta" / "episode_index_to_id.json"
    with open(id_map_path) as f:
        episode_id_map = json.load(f)  # {str(ep_idx): {canonical_id: ...}}

    # Build dataset (same config as caching run)
    embodiment_tag = EmbodimentTag[args.embodiment_tag]
    modality_config = MODALITY_CONFIGS[args.embodiment_tag.lower()]

    allowed_episode_indices = None
    if args.labs:
        allowed_episode_indices = get_allowed_episode_indices(args.dataset_path, args.labs)
        print(f"[rank {args.rank}] Lab filter {args.labs}: {len(allowed_episode_indices)} episodes")

    base_dataset = ShardedSingleStepDataset(
        dataset_path=args.dataset_path,
        embodiment_tag=embodiment_tag,
        modality_configs=modality_config,
        shard_size=args.shard_size,
        episode_sampling_rate=args.episode_sampling_rate,
        seed=42,
        allow_padding=False,
        allowed_episode_indices=allowed_episode_indices,
    )
    total_shards = len(base_dataset)
    print(f"[rank {args.rank}] Dataset: {total_shards} shards")

    # Load backbone cache index to get shard seq_lens
    cache_index = torch.load(Path(args.cache_dir) / "index.pt", weights_only=True)

    # Static strided shard assignment (matches backbone caching)
    my_shards = range(args.rank, total_shards, args.world_size)
    print(f"[rank {args.rank}] Processing {len(my_shards)} shards")

    for shard_idx in tqdm(my_shards, desc=f"Shards [rank {args.rank}]"):
        cache_path = pos_cache_dir / f"pos_shard_{shard_idx:04d}.pt"
        done_path  = pos_cache_dir / f"pos_shard_{shard_idx:04d}.done"

        if done_path.exists():
            continue

        # Load the backbone shard to get seq_len (positions must match)
        backbone_shard = torch.load(
            Path(args.cache_dir) / f"shard_{shard_idx:05d}.pt",
            weights_only=True,
            map_location="cpu",
        )
        shard_seq_len = backbone_shard["backbone_features"].shape[1]
        n_samples = backbone_shard["backbone_features"].shape[0]

        datapoints = base_dataset.get_shard(shard_idx)
        assert len(datapoints) == n_samples, (
            f"Shard {shard_idx}: datapoints={len(datapoints)} != backbone={n_samples}"
        )

        all_positions = []
        for dp_idx, dp in enumerate(datapoints):
            image_mask = backbone_shard["image_mask"][dp_idx].numpy()  # [shard_seq_len] bool

            # Get episode_id and frame index from the datapoint metadata
            episode_idx = dp.get("episode_index", None)
            frame_idx = dp.get("frame_index", None)
            episode_meta = episode_id_map.get(str(episode_idx), {})
            episode_id = episode_meta.get("canonical_id", None)

            # Get image_grid_thw — stored by the backbone cache script via image_meta
            # Fall back to zeros if not available (depth not yet extracted)
            image_grid_thw = dp.get("image_grid_thw", None)

            depth_path = None
            intrinsics = None
            T_cam2base = None

            if episode_id is not None and frame_idx is not None and image_grid_thw is not None:
                depth_path = (
                    Path(args.depth_dir) / episode_id / f"depth_{frame_idx:06d}.npy"
                )
                intrinsics = all_intrinsics.get(episode_id, None)
                ep_extrinsics = all_extrinsics.get(episode_id, {})
                T_cam2base = ep_extrinsics.get(str(frame_idx), None)

                if isinstance(image_grid_thw, (list, np.ndarray)):
                    image_grid_thw = np.array(image_grid_thw)
            else:
                image_grid_thw = np.zeros((0, 3), dtype=np.int64)

            positions = compute_positions_for_sample(
                image_grid_thw=image_grid_thw,
                image_mask=image_mask[:image_mask.sum()],  # trim to actual seq_len
                depth_path=depth_path,
                intrinsics=intrinsics,
                T_cam2base=T_cam2base,
            )

            # Pad to shard_seq_len to match backbone shard layout
            if len(positions) < shard_seq_len:
                pad = np.zeros((shard_seq_len - len(positions), 3), dtype=np.float32)
                positions = np.concatenate([positions, pad], axis=0)
            else:
                positions = positions[:shard_seq_len]

            all_positions.append(positions)

        shard_data = {
            "token_positions_3d": torch.from_numpy(np.stack(all_positions))  # [N, seq_len, 3]
        }

        tmp_path = cache_path.with_suffix(".tmp")
        torch.save(shard_data, tmp_path)
        tmp_path.rename(cache_path)
        done_path.touch()
        print(f"[rank {args.rank}] Saved shard {shard_idx}: {n_samples} samples, "
              f"seq_len={shard_seq_len}")

    print(f"\n[rank {args.rank}] Done!")


if __name__ == "__main__":
    main()
