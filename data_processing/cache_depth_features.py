"""
cache_depth_features.py

Pre-computes token_positions_3d for every cached sample and saves them as
depth shard files alongside the backbone cache.

For each backbone cache shard, this script:
  1. Groups samples by episode (for efficient depth I/O — one load per episode)
  2. Loads depth.blosc + intrinsics.npy + extrinsics per episode
  3. Unprojects patches → world-frame (x,y,z) per vision token
  4. Saves depth_shard_XXXXX.pt = {"token_positions_3d": [N, seq_len, 3]}

Run as a SLURM array job:
    sbatch --array=0-3 scripts/slurm/cache_depth_features.slurm

Static strided shard assignment (no locks):
    rank 0 → shards 0, 4, 8, ...
    rank 1 → shards 1, 5, 9, ...

Reads:
    {backbone_cache_dir}/index.pt                          — global_idx → (shard_idx, row)
    {backbone_cache_dir}/shard_XXXXX.pt                    — for image_mask
    {depth_dir}/episode_frame_index.pkl                    — global_idx → {canonical_id, frame_idx}
    {depth_dir}/serial_map.json                            — canonical_id → {ext1, wrist}
    {depth_dir}/valid_canonical_ids.json                   — set of valid episodes
    {depth_dir}/{canonical_id}/{serial}/depth.blosc        — (T, H, W) float32
    {depth_dir}/{canonical_id}/{serial}/shape.npy          — (3,) int
    {depth_dir}/{canonical_id}/{serial}/intrinsics.npy     — [fx, fy, cx, cy]
    {raw_dir}/{canonical_id}/metadata_*.json               — ext1 cam2base extrinsics
    {raw_dir}/{canonical_id}/trajectory.h5                 — wrist per-timestep extrinsics

Writes:
    {backbone_cache_dir}/depth_shard_XXXXX.pt              — {token_positions_3d: [N, seq_len, 3]}
    {backbone_cache_dir}/depth_shard_XXXXX.done            — completion sentinel
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import blosc
import cv2
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# ── DROID / Eagle constants (must match processing_gr00t_n1d6.py) ────────────
_RESIZED_H    = 168
_RESIZED_W    = 308
_TOKEN_STRIDE = 28
_GRID_ROWS    = 6
_GRID_COLS    = 11   # tokens per camera = 6 × 11 = 66


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_depth_episode(depth_dir: Path, canonical_id: str, serial: str) -> np.ndarray:
    """Load (T, H, W) float32 depth array from depth.blosc."""
    ep_dir = depth_dir / canonical_id / serial
    shape = np.load(ep_dir / "shape.npy")
    raw = (ep_dir / "depth.blosc").read_bytes()
    return np.frombuffer(blosc.decompress(raw), dtype=np.float32).reshape(shape)


def load_intrinsics(depth_dir: Path, canonical_id: str, serial: str) -> np.ndarray:
    """Load [fx, fy, cx, cy] intrinsics at 180×320."""
    return np.load(depth_dir / canonical_id / serial / "intrinsics.npy")


def get_ext1_cam2base(raw_dir: Path, canonical_id: str) -> np.ndarray | None:
    """Static 4×4 cam-to-base for exterior camera from metadata JSON.

    Convention: cam→base (T_cam2base). Rotation is Euler XYZ (not rotvec).
    """
    ep_dir = raw_dir / canonical_id
    meta_files = list(ep_dir.glob("metadata_*.json")) if ep_dir.exists() else []
    if not meta_files:
        return None
    meta = json.loads(meta_files[0].read_text())
    dof = meta.get("ext1_cam_extrinsics")
    if dof is None:
        return None
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler("xyz", dof[3:]).as_matrix()
    T[:3, 3] = dof[:3]
    return T


def get_ext1_cam2base_cam2cam(cam2cam: dict, canonical_id: str) -> np.ndarray | None:
    """Static 4×4 cam-to-base for ext1 (left_cam) from cam2cam_extrinsics.json.

    cam2cam stores calibrated poses as 4×4 cam→world (robot base) matrices.
    left_cam = ext1 (confirmed by focal-length matching against intrinsics.npy).
    """
    ep = cam2cam.get(canonical_id)
    if ep is None:
        return None
    left = ep.get("left_cam")
    if left is None:
        return None
    pose = left.get("pose")
    if pose is None:
        return None
    return np.array(pose, dtype=np.float64)


def get_ext2_cam2base(raw_dir: Path, canonical_id: str) -> np.ndarray | None:
    """Static 4×4 cam-to-base for ext2 from metadata JSON (Euler XYZ, cam→base)."""
    ep_dir = raw_dir / canonical_id
    meta_files = list(ep_dir.glob("metadata_*.json")) if ep_dir.exists() else []
    if not meta_files:
        return None
    meta = json.loads(meta_files[0].read_text())
    dof = meta.get("ext2_cam_extrinsics")
    if dof is None:
        return None
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler("xyz", dof[3:]).as_matrix()
    T[:3, 3] = dof[:3]
    return T


def get_ext2_cam2base_cam2cam(cam2cam: dict, canonical_id: str) -> np.ndarray | None:
    """Static 4×4 cam-to-base for ext2 (right_cam) from cam2cam_extrinsics.json."""
    ep = cam2cam.get(canonical_id)
    if ep is None:
        return None
    right = ep.get("right_cam")
    if right is None:
        return None
    pose = right.get("pose")
    if pose is None:
        return None
    return np.array(pose, dtype=np.float64)


def get_wrist_extrinsics_array(raw_dir: Path, canonical_id: str, wrist_serial: str) -> np.ndarray | None:
    """Load full (T, 6) wrist extrinsics array from trajectory.h5."""
    traj_path = raw_dir / canonical_id / "trajectory.h5"
    if not traj_path.exists():
        return None
    key = f"observation/camera_extrinsics/{wrist_serial}_left"
    with h5py.File(traj_path, "r") as f:
        if key not in f:
            return None
        return f[key][:]  # (T, 6) float64


def get_eef_positions_array(raw_dir: Path, canonical_id: str) -> np.ndarray | None:
    """Load (T, 3) EEF xyz positions in robot base frame from trajectory.h5.

    Source: observation/robot_state/cartesian_position[:, :3]
    Same coordinate frame as depth-unprojected image token positions.
    Used to compute relative positions p_k - p_eef for RoPE.
    """
    traj_path = raw_dir / canonical_id / "trajectory.h5"
    if not traj_path.exists():
        return None
    with h5py.File(traj_path, "r") as f:
        key = "observation/robot_state/cartesian_position"
        if key not in f:
            return None
        return f[key][:, :3].astype(np.float32)  # (T, 3)


def dof_to_mat(dof: np.ndarray) -> np.ndarray:
    """Convert [tx, ty, tz, rx, ry, rz] to 4×4 cam→base transform.

    Rotation is Euler XYZ (intrinsic, radians) — NOT rotvec/axis-angle.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler("xyz", dof[3:]).as_matrix()
    T[:3, 3] = dof[:3]
    return T


# ── Core unprojection ─────────────────────────────────────────────────────────

def unproject_patches(
    depth_frame: np.ndarray,     # (H_orig, W_orig) float32, metres
    intrinsics: np.ndarray,      # [fx, fy, cx, cy] at original resolution
    T_cam2base: np.ndarray,      # (4, 4) float64
    token_seq_positions: np.ndarray,  # indices into seq for this camera's tokens
    positions: np.ndarray,       # (seq_len, 3) output — modified in-place
) -> None:
    """Unproject each patch's valid depth pixels to world-frame average position."""
    depth_resized = cv2.resize(depth_frame, (_RESIZED_W, _RESIZED_H),
                               interpolation=cv2.INTER_NEAREST)

    orig_h, orig_w = depth_frame.shape
    fx180, fy180, cx180, cy180 = intrinsics
    fx = fx180 * _RESIZED_W / orig_w
    fy = fy180 * _RESIZED_H / orig_h
    cx = cx180 * _RESIZED_W / orig_w
    cy = cy180 * _RESIZED_H / orig_h

    for t, seq_pos in enumerate(token_seq_positions):
        row = t // _GRID_COLS
        col = t % _GRID_COLS
        y0, y1 = row * _TOKEN_STRIDE, (row + 1) * _TOKEN_STRIDE
        x0, x1 = col * _TOKEN_STRIDE, (col + 1) * _TOKEN_STRIDE

        patch_depth = depth_resized[y0:y1, x0:x1]
        valid = np.isfinite(patch_depth) & (patch_depth > 0)
        if not valid.any():
            continue

        ys_all = np.arange(y0, y1, dtype=np.float32)
        xs_all = np.arange(x0, x1, dtype=np.float32)
        Y_all, X_all = np.meshgrid(ys_all, xs_all, indexing="ij")
        D_v = patch_depth[valid]
        X_cam = (X_all[valid] - cx) * D_v / fx
        Y_cam = (Y_all[valid] - cy) * D_v / fy

        pts = np.stack([X_cam, Y_cam, D_v, np.ones_like(D_v)], axis=-1)  # (N, 4)
        pts_world = pts @ T_cam2base.T  # (N, 4)
        positions[seq_pos] = pts_world[:, :3].mean(axis=0)


# ── Per-episode compute ────────────────────────────────────────────────────────

def compute_positions_for_episode(
    canonical_id: str,
    frame_entries: list[tuple[int, int]],  # [(row_in_shard, frame_idx), ...]
    image_masks: np.ndarray,               # (N, seq_len) bool
    depth_dir: Path,
    raw_dir: Path,
    serial_map: dict,
    valid_ids: set | None,
    cam2cam: dict | None = None,
    use_ext2: bool = False,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Compute token_positions_3d and eef_position_3d for all frames of one episode.
    Returns:
        token_results: {row_in_shard: positions [seq_len, 3]}
        eef_results:   {row_in_shard: eef_xyz [3]}
    Zero positions = identity RoPE (used for invalid episodes / missing data).
    """
    token_results = {}
    eef_results = {}
    cam_pos_results = {}  # {row: np.ndarray [num_cameras, 3]}

    def zeros_for_row(row):
        seq_len = image_masks[row].shape[0]
        return np.zeros((seq_len, 3), dtype=np.float32)

    num_cameras = 2  # ext1 + (ext2 or wrist)

    if valid_ids is not None and canonical_id not in valid_ids:
        for row, _ in frame_entries:
            token_results[row] = zeros_for_row(row)
            eef_results[row] = np.zeros(3, dtype=np.float32)
            cam_pos_results[row] = np.zeros((num_cameras, 3), dtype=np.float32)
        return token_results, eef_results, cam_pos_results

    serials = serial_map.get(canonical_id)
    if serials is None:
        for row, _ in frame_entries:
            token_results[row] = zeros_for_row(row)
            eef_results[row] = np.zeros(3, dtype=np.float32)
            cam_pos_results[row] = np.zeros((num_cameras, 3), dtype=np.float32)
        return token_results, eef_results, cam_pos_results

    ext1_serial  = serials["ext1"]
    cam2_serial  = serials["ext2"] if use_ext2 else serials["wrist"]

    # Load per-episode data once
    T_ext1 = None
    try:
        depth_ext1   = load_depth_episode(depth_dir, canonical_id, ext1_serial)
        intr_ext1    = load_intrinsics(depth_dir, canonical_id, ext1_serial)
        if cam2cam is not None:
            T_ext1 = get_ext1_cam2base_cam2cam(cam2cam, canonical_id)
        else:
            T_ext1 = get_ext1_cam2base(raw_dir, canonical_id)
    except Exception:
        depth_ext1 = None

    T_cam2_static = None
    cam2_dofs = None
    try:
        depth_cam2  = load_depth_episode(depth_dir, canonical_id, cam2_serial)
        intr_cam2   = load_intrinsics(depth_dir, canonical_id, cam2_serial)
        if use_ext2:
            # ext2: static extrinsics (from cam2cam or metadata)
            if cam2cam is not None:
                T_cam2_static = get_ext2_cam2base_cam2cam(cam2cam, canonical_id)
            else:
                T_cam2_static = get_ext2_cam2base(raw_dir, canonical_id)
        else:
            # wrist: per-timestep extrinsics from trajectory.h5
            cam2_dofs = get_wrist_extrinsics_array(raw_dir, canonical_id, cam2_serial)
    except Exception:
        depth_cam2 = None

    eef_dofs = get_eef_positions_array(raw_dir, canonical_id)  # (T, 3) or None

    # Extract camera optical centers from extrinsics (T_cam2base[:3, 3] = camera origin in base frame)
    # These are static per-episode for ext1/ext2 (from cam2cam), or per-timestep for wrist.
    ext1_optical_center = T_ext1[:3, 3].astype(np.float32) if T_ext1 is not None else np.zeros(3, dtype=np.float32)
    if use_ext2:
        # ext2 has static extrinsics
        cam2_optical_center_static = T_cam2_static[:3, 3].astype(np.float32) if T_cam2_static is not None else np.zeros(3, dtype=np.float32)
    else:
        cam2_optical_center_static = None  # wrist is per-timestep

    for row, frame_idx in frame_entries:
        mask = image_masks[row]  # (seq_len,) bool
        seq_len = mask.shape[0]
        positions = np.zeros((seq_len, 3), dtype=np.float32)

        image_pos = np.where(mask)[0]
        n_per_cam = len(image_pos) // 2
        ext1_seq  = image_pos[:n_per_cam]
        cam2_seq  = image_pos[n_per_cam:]

        # Exterior camera 1 (static extrinsics)
        if depth_ext1 is not None and T_ext1 is not None:
            fi = min(frame_idx, depth_ext1.shape[0] - 1)
            try:
                unproject_patches(depth_ext1[fi], intr_ext1, T_ext1, ext1_seq, positions)
            except Exception:
                pass

        # Camera 2: ext2 (static) or wrist (per-timestep)
        T_cam2 = None
        if depth_cam2 is not None:
            fi = min(frame_idx, depth_cam2.shape[0] - 1)
            if use_ext2:
                T_cam2 = T_cam2_static
            else:
                if cam2_dofs is not None:
                    fi_dof = min(frame_idx, len(cam2_dofs) - 1)
                    T_cam2 = dof_to_mat(cam2_dofs[fi_dof])
            if T_cam2 is not None:
                try:
                    unproject_patches(depth_cam2[fi], intr_cam2, T_cam2, cam2_seq, positions)
                except Exception:
                    pass

        token_results[row] = positions

        # EEF position in robot base frame
        if eef_dofs is not None:
            fi_eef = min(frame_idx, eef_dofs.shape[0] - 1)
            eef_results[row] = eef_dofs[fi_eef]
        else:
            eef_results[row] = np.zeros(3, dtype=np.float32)

        # Camera optical centers [num_cameras, 3]
        cam_pos = np.zeros((num_cameras, 3), dtype=np.float32)
        cam_pos[0] = ext1_optical_center
        if use_ext2:
            cam_pos[1] = cam2_optical_center_static
        else:
            # Wrist: extract from per-timestep extrinsics
            if T_cam2 is not None:
                cam_pos[1] = T_cam2[:3, 3].astype(np.float32)
        cam_pos_results[row] = cam_pos

    return token_results, eef_results, cam_pos_results


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone-cache-dir", type=str, required=True,
                   help="Path to backbone cache (contains index.pt, shard_XXXXX.pt)")
    p.add_argument("--depth-dir", type=str, required=True,
                   help="Path to depth cache (contains episode_frame_index.pkl, serial_map.json)")
    p.add_argument("--raw-dir", type=str, required=True,
                   help="Path to raw DROID episodes (contains metadata_*.json, trajectory.h5)")
    p.add_argument("--cam2cam-json", type=str, default=None,
                   help="If set, use calibrated ext1 extrinsics from cam2cam_extrinsics.json "
                        "instead of per-episode metadata_*.json")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory to write depth_shard_*.pt files. "
                        "Defaults to --backbone-cache-dir if not set.")
    p.add_argument("--use-ext2", action="store_true",
                   help="Use ext2 as the second camera instead of wrist. "
                        "Requires ext2 serial in serial_map.json and ext2 depth data.")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    backbone_dir = Path(args.backbone_cache_dir)
    depth_dir    = Path(args.depth_dir)
    raw_dir      = Path(args.raw_dir)
    output_dir   = Path(args.output_dir) if args.output_dir else backbone_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cam2cam = None
    if args.cam2cam_json:
        print(f"[rank {args.rank}] Loading cam2cam_extrinsics.json ...")
        with open(args.cam2cam_json) as f:
            cam2cam = json.load(f)
        print(f"[rank {args.rank}] cam2cam loaded ({len(cam2cam)} episodes)")
    else:
        print(f"[rank {args.rank}] Using metadata_*.json for extrinsics")

    use_ext2 = args.use_ext2
    print(f"[rank {args.rank}] Second camera: {'ext2 (static)' if use_ext2 else 'wrist (per-timestep)'}")

    # ── Load index and metadata ───────────────────────────────────────────────
    print(f"[rank {args.rank}] Loading index.pt ...")
    index = torch.load(backbone_dir / "index.pt", weights_only=True)
    shard_idx_of = index["shard_idx"]  # (N,) int32
    row_of       = index["row"]        # (N,) int32
    total_samples = len(shard_idx_of)
    total_shards  = int(shard_idx_of.max().item()) + 1
    print(f"[rank {args.rank}] {total_samples} samples, {total_shards} shards")

    print(f"[rank {args.rank}] Loading episode_frame_index.pkl ...")
    with open(depth_dir / "episode_frame_index.pkl", "rb") as f:
        episode_frame_index = pickle.load(f)  # list[{canonical_id, frame_idx}]

    with open(depth_dir / "serial_map.json") as f:
        serial_map = json.load(f)

    valid_ids = None
    valid_ids_path = depth_dir / "valid_canonical_ids.json"
    if valid_ids_path.exists():
        with open(valid_ids_path) as f:
            valid_ids = set(json.load(f))
        print(f"[rank {args.rank}] {len(valid_ids)} valid episodes loaded")

    # ── Build per-shard sample lists (global_idx grouped by shard) ───────────
    # shard_samples[shard_idx] = [(global_idx, row_in_shard), ...]
    shard_samples: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for global_idx in range(total_samples):
        s = int(shard_idx_of[global_idx])
        r = int(row_of[global_idx])
        shard_samples[s].append((global_idx, r))

    # ── Process assigned shards ───────────────────────────────────────────────
    my_shards = range(args.rank, total_shards, args.world_size)
    print(f"[rank {args.rank}] Processing {len(my_shards)} shards "
          f"(stride {args.world_size}, starting at {args.rank})")

    for shard_idx in tqdm(my_shards, desc=f"Depth shards [rank {args.rank}]"):
        out_path  = output_dir / f"depth_shard_{shard_idx:05d}.pt"
        done_path = output_dir / f"depth_shard_{shard_idx:05d}.done"

        if done_path.exists():
            continue

        # Load image_mask for this shard (needed to find image token positions)
        shard_data = torch.load(
            backbone_dir / f"shard_{shard_idx:05d}.pt",
            weights_only=True, map_location="cpu"
        )
        image_masks = shard_data["image_mask"].numpy()  # (N, seq_len) bool
        shard_size  = image_masks.shape[0]
        seq_len     = image_masks.shape[1]
        del shard_data

        # Group samples in this shard by episode
        samples = shard_samples[shard_idx]  # [(global_idx, row_in_shard)]
        by_episode: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for global_idx, row in samples:
            entry = episode_frame_index[global_idx]
            by_episode[entry["canonical_id"]].append((row, entry["frame_idx"]))

        # Allocate output tensors
        all_positions    = np.zeros((shard_size, seq_len, 3), dtype=np.float32)
        all_eef          = np.zeros((shard_size, 3),           dtype=np.float32)
        all_camera_pos   = np.zeros((shard_size, 2, 3),        dtype=np.float32)

        # Process episode by episode (one depth load per episode)
        for canonical_id, frame_entries in by_episode.items():
            token_results, eef_results, cam_pos_results = compute_positions_for_episode(
                canonical_id, frame_entries, image_masks,
                depth_dir, raw_dir, serial_map, valid_ids,
                cam2cam=cam2cam,
                use_ext2=use_ext2,
            )
            for row, pos in token_results.items():
                all_positions[row, :pos.shape[0]] = pos
            for row, eef in eef_results.items():
                all_eef[row] = eef
            for row, cam_pos in cam_pos_results.items():
                all_camera_pos[row] = cam_pos

        # Save
        result = {
            "token_positions_3d":  torch.from_numpy(all_positions),
            "eef_position_3d":     torch.from_numpy(all_eef),
            "camera_positions_3d": torch.from_numpy(all_camera_pos),
        }
        tmp_path = out_path.with_suffix(".tmp")
        torch.save(result, tmp_path)
        tmp_path.rename(out_path)
        done_path.touch()

    print(f"[rank {args.rank}] Done.")


if __name__ == "__main__":
    main()
