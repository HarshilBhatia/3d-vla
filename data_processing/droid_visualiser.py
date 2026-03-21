"""
droid_visualiser.py

Per-rollout 3D visualiser for the DROID dataset using Rerun.

For a given episode:
  - Shows 2D images from all 3 cameras (ext1, ext2, wrist) on a time slider
  - Unprojects depth.blosc files → coloured 3D point clouds in world frame
  - Logs robot EEF position and language annotation

Usage:
    python data_processing/droid_visualiser.py \
        --dataset-dir /work/nvme/bgkz/droid_raw_large_superset \
        --raw-dir /work/nvme/bgkz/droid_multilab_raw \
        --depth-dir /work/nvme/bgkz/droid_multilab_depths \
        --episode-idx 17000 \
        --max-timesteps 30 \
        --serve
"""

import argparse
import json
import warnings
from pathlib import Path

import av
import blosc
import cv2
import h5py
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation

# ── Constants ─────────────────────────────────────────────────────────────────
_DEPTH_H = 180
_DEPTH_W = 320
_MAX_CLOUD_PTS = 10_000


# ── Verbatim copies from cache_depth_features.py ─────────────────────────────

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

    Convention: cam→base (T_base_cam). Rotation is Euler XYZ (not rotvec).
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


def get_ext2_cam2base_and_serial(raw_dir: Path, canonical_id: str) -> tuple[str, np.ndarray] | None:
    """Return (ext2_serial, T_ext2_cam2base) from raw metadata JSON.

    Convention: cam→base (T_base_cam). Rotation is Euler XYZ (not rotvec).
    """
    ep_dir = raw_dir / canonical_id
    meta_files = list(ep_dir.glob("metadata_*.json")) if ep_dir.exists() else []
    if not meta_files:
        return None
    meta = json.loads(meta_files[0].read_text())
    serial = meta.get("ext2_cam_serial")
    dof = meta.get("ext2_cam_extrinsics")
    if serial is None or dof is None:
        return None
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler("xyz", dof[3:]).as_matrix()
    T[:3, 3] = dof[:3]
    return str(serial), T


def get_wrist_extrinsics_array(raw_dir: Path, canonical_id: str, wrist_serial: str) -> np.ndarray | None:
    """Load full (T, 6) wrist extrinsics array from trajectory.h5."""
    return get_camera_extrinsics_array(raw_dir, canonical_id, wrist_serial)


def get_camera_extrinsics_array(raw_dir: Path, canonical_id: str, camera_serial: str) -> np.ndarray | None:
    """Load (T, 6) camera extrinsics array from trajectory.h5.

    Expected dataset key pattern: observation/camera_extrinsics/{camera_serial}_left
    """
    traj_path = raw_dir / canonical_id / "trajectory.h5"
    if not traj_path.exists():
        return None
    key = f"observation/camera_extrinsics/{camera_serial}_left"
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

    Rotation is Euler XYZ (intrinsic, radians) — used for wrist extrinsics
    from trajectory.h5. NOT rotvec/axis-angle.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler("xyz", dof[3:]).as_matrix()
    T[:3, 3] = dof[:3]
    return T


# ── Dense unprojection for visualisation ─────────────────────────────────────

def unproject_depth_frame(
    depth_frame: np.ndarray,   # (H_orig, W_orig) float32, metres
    intrinsics: np.ndarray,    # [fx, fy, cx, cy] at 180×320
    T_cam2world: np.ndarray,   # (4, 4) float64
    rgb_frame: np.ndarray,     # (H_img, W_img, 3) uint8 — for point colours (required)
    max_pts: int = _MAX_CLOUD_PTS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unproject all valid depth pixels → world-frame XYZ, with corresponding RGB.

    Returns:
        pts_world : (N, 3) float32
        colors    : (N, 3) uint8
    """
    depth_resized = cv2.resize(depth_frame, (_DEPTH_W, _DEPTH_H),
                               interpolation=cv2.INTER_NEAREST)

    fx, fy, cx, cy = intrinsics

    valid = np.isfinite(depth_resized) & (depth_resized > 0.05) & (depth_resized < 2.0)
    ys, xs = np.where(valid)
    D = depth_resized[ys, xs]

    X_cam = (xs - cx) * D / fx
    Y_cam = (ys - cy) * D / fy
    ones  = np.ones_like(D)
    pts_cam = np.stack([X_cam, Y_cam, D, ones], axis=-1)  # (N, 4)

    pts_world = (pts_cam @ T_cam2world.T)[:, :3].astype(np.float32)
    rgb_resized = cv2.resize(rgb_frame, (_DEPTH_W, _DEPTH_H),
                             interpolation=cv2.INTER_LINEAR)
    colors = rgb_resized[ys, xs]  # (N, 3) uint8

    # Subsample
    if len(pts_world) > max_pts:
        idx = np.random.choice(len(pts_world), max_pts, replace=False)
        pts_world = pts_world[idx]
        colors    = colors[idx]

    return pts_world, colors


# ── Image loading via PyAV ────────────────────────────────────────────────────

def decode_frames_at(mp4_path: Path, frame_indices: np.ndarray) -> dict[int, np.ndarray]:
    """
    Decode only the frames at the given indices from an mp4.

    Returns dict {frame_idx: (H, W, 3) uint8}.
    """
    target = set(frame_indices.tolist())
    frames = {}
    with av.open(str(mp4_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        i = 0
        for packet in container.demux(stream):
            for frame in packet.decode():
                if i in target:
                    frames[i] = frame.to_ndarray(format="rgb24")
                    target.discard(i)
                    if not target:
                        return frames
                i += 1
    return frames


# ── Episode resolution helpers ────────────────────────────────────────────────

def resolve_episode(
    dataset_dir: Path,
    episode_idx: int | None,
    canonical_id: str | None,
) -> tuple[int, str]:
    """
    Return (episode_idx, canonical_id).  Exactly one of the inputs must be set.
    """
    if episode_idx is not None and canonical_id is not None:
        raise ValueError("Provide --episode-idx OR --canonical-id, not both.")
    if episode_idx is None and canonical_id is None:
        raise ValueError("Provide exactly one of --episode-idx or --canonical-id.")

    id_map_path = dataset_dir / "meta" / "episode_index_to_id.json"
    if not id_map_path.exists():
        raise ValueError(f"episode_index_to_id.json not found at {id_map_path}")
    with open(id_map_path) as f:
        id_map = json.load(f)  # {str(idx): canonical_id}

    # Values are dicts: {"canonical_id": "LAB+...", "stored_id": "gs://..."}
    # Normalise to str canonical_id
    def _extract_cid(v):
        return v["canonical_id"] if isinstance(v, dict) else v

    if canonical_id is not None:
        reverse = {_extract_cid(v): int(k) for k, v in id_map.items()}
        if canonical_id not in reverse:
            raise ValueError(f"canonical_id {canonical_id!r} not found in episode_index_to_id.json")
        return reverse[canonical_id], canonical_id

    # episode_idx provided
    key = str(episode_idx)
    if key not in id_map:
        raise ValueError(f"episode_idx {episode_idx} not found in episode_index_to_id.json")
    return episode_idx, _extract_cid(id_map[key])


def get_episode_meta(dataset_dir: Path, episode_idx: int) -> dict:
    """Read the episodes.jsonl entry with episode_index == episode_idx."""
    jsonl_path = dataset_dir / "meta" / "episodes.jsonl"
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("episode_index") == episode_idx:
                return d
    raise ValueError(f"episode_idx {episode_idx} not found in episodes.jsonl")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DROID per-rollout 3D visualiser (Rerun)")
    p.add_argument("--dataset-dir", type=str,
                   default="/work/nvme/bgkz/droid_raw_large_superset",
                   help="Path to droid_raw_large_superset")
    p.add_argument("--raw-dir", type=str, required=True,
                   help="Path to raw episodes (metadata_*.json + trajectory.h5)")
    p.add_argument("--depth-dir", type=str, required=True,
                   help="Path to depth cache ({canonical_id}/{serial}/depth.blosc + intrinsics.npy)")
    p.add_argument("--episode-idx", type=int, default=None,
                   help="Integer episode index (mutually exclusive with --canonical-id)")
    p.add_argument("--canonical-id", type=str, default=None,
                   help="e.g. RAIL+abc123+2023-10-01-12h (mutually exclusive with --episode-idx)")
    p.add_argument("--max-timesteps", type=int, default=30,
                   help="Max frames to visualise (default: 30)")
    p.add_argument("--serve", action="store_true",
                   help="If set, serve for live browser view; otherwise saves to {canonical_id}.rrd")
    p.add_argument("--grpc-port", type=int, default=9876,
                   help="gRPC server port (default: 9876)")
    p.add_argument("--web-port", type=int, default=9090,
                   help="Web viewer port (default: 9090)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.raw_dir is None:
        raise ValueError("--raw-dir is required")
    if args.depth_dir is None:
        raise ValueError("--depth-dir is required")

    dataset_dir = Path(args.dataset_dir)
    raw_dir     = Path(args.raw_dir)
    depth_dir   = Path(args.depth_dir)

    # ── Resolve episode ───────────────────────────────────────────────────────
    episode_idx, canonical_id = resolve_episode(dataset_dir, args.episode_idx, args.canonical_id)
    print(f"Episode index : {episode_idx}")
    print(f"Canonical ID  : {canonical_id}")

    # ── Episode metadata ──────────────────────────────────────────────────────
    ep_meta = get_episode_meta(dataset_dir, episode_idx)
    length  = ep_meta.get("length") or ep_meta.get("episode_length")
    if length is None:
        raise ValueError(f"Could not determine episode length from episodes.jsonl entry: {ep_meta}")
    tasks_raw = ep_meta.get("tasks") or ep_meta.get("language_instruction") or ""
    if isinstance(tasks_raw, list):
        task_text = " | ".join(t for t in tasks_raw if t)
    else:
        task_text = str(tasks_raw)
    print(f"Episode length: {length} frames")
    print(f"Task          : {task_text[:120]}")

    # ── Frame subsampling ─────────────────────────────────────────────────────
    frame_indices = np.linspace(0, length - 1, min(args.max_timesteps, length), dtype=int)
    print(f"Visualising {len(frame_indices)} frames: {frame_indices}")

    # ── Chunk for video/parquet paths ─────────────────────────────────────────
    chunk = episode_idx // 1000

    # ── Load serial map ───────────────────────────────────────────────────────
    serial_map_path = depth_dir / "serial_map.json"
    if not serial_map_path.exists():
        raise ValueError(f"serial_map.json not found at {serial_map_path}")
    with open(serial_map_path) as f:
        serial_map = json.load(f)

    serials = serial_map.get(canonical_id)
    if serials is None:
        warnings.warn(f"canonical_id {canonical_id!r} not in serial_map.json — depth will be skipped")

    ext1_serial   = serials["ext1"]   if serials else None
    wrist_serial  = serials["wrist"]  if serials else None
    ext2_serial   = serials.get("ext2") if serials else None

    # serial_map.json in this codebase only contains ext1 + wrist.
    # For ext2 we fall back to raw metadata (ext2_cam_serial + ext2_cam_extrinsics).
    T_ext2 = None
    ext2_meta = get_ext2_cam2base_and_serial(raw_dir, canonical_id)
    if ext2_meta is not None:
        ext2_serial_meta, T_ext2 = ext2_meta
        if ext2_serial is None:
            ext2_serial = ext2_serial_meta

    # ── Load depth arrays (one per camera, whole episode) ────────────────────
    depth_ext1  = depth_intr_ext1  = None
    depth_wrist = depth_intr_wrist = None
    depth_ext2  = depth_intr_ext2  = None

    if ext1_serial:
        try:
            depth_ext1      = load_depth_episode(depth_dir, canonical_id, ext1_serial)
            depth_intr_ext1 = load_intrinsics(depth_dir, canonical_id, ext1_serial)
            print(f"Loaded ext1  depth: {depth_ext1.shape}")
        except Exception as e:
            warnings.warn(f"Could not load ext1 depth: {e}")

    if wrist_serial:
        try:
            depth_wrist      = load_depth_episode(depth_dir, canonical_id, wrist_serial)
            depth_intr_wrist = load_intrinsics(depth_dir, canonical_id, wrist_serial)
            print(f"Loaded wrist depth: {depth_wrist.shape}")
        except Exception as e:
            warnings.warn(f"Could not load wrist depth: {e}")

    if ext2_serial:
        try:
            depth_ext2      = load_depth_episode(depth_dir, canonical_id, ext2_serial)
            depth_intr_ext2 = load_intrinsics(depth_dir, canonical_id, ext2_serial)
            print(f"Loaded ext2  depth: {depth_ext2.shape}")
        except Exception as e:
            warnings.warn(f"Could not load ext2 depth: {e}")

    # ── Load extrinsics ───────────────────────────────────────────────────────
    T_ext1 = get_ext1_cam2base(raw_dir, canonical_id)
    if T_ext1 is None:
        warnings.warn("ext1 extrinsics not found — ext1 depth cloud will use identity transform")
        T_ext1 = np.eye(4, dtype=np.float64)

    wrist_dofs = None
    if wrist_serial:
        wrist_dofs = get_wrist_extrinsics_array(raw_dir, canonical_id, wrist_serial)
        if wrist_dofs is None:
            warnings.warn("Wrist extrinsics not found — wrist depth cloud will use identity transform")

    eef_positions = get_eef_positions_array(raw_dir, canonical_id)  # (T, 3) or None
    if eef_positions is None:
        warnings.warn("EEF positions not found — EEF marker will be skipped")

    # ── Camera name → mp4 path ────────────────────────────────────────────────
    cam_names = [
        "exterior_image_1_left",
        "exterior_image_2_left",
        "wrist_image_left",
    ]
    cam_mp4: dict[str, Path] = {}
    for cam in cam_names:
        mp4 = (dataset_dir / "videos" / f"chunk-{chunk:03d}"
               / f"observation.images.{cam}" / f"episode_{episode_idx:06d}.mp4")
        if mp4.exists():
            cam_mp4[cam] = mp4
        else:
            print(f"  [skip] mp4 not found: {mp4}")

    # ── Decode all needed frames ──────────────────────────────────────────────
    cam_frames: dict[str, dict[int, np.ndarray]] = {}
    for cam, mp4 in cam_mp4.items():
        print(f"Decoding {cam} ...")
        try:
            cam_frames[cam] = decode_frames_at(mp4, frame_indices)
        except Exception as e:
            warnings.warn(f"Failed to decode {cam}: {e}")

    # ── Rerun init + blueprint ────────────────────────────────────────────────
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="3D world", origin="world"),
            rrb.Vertical(
                rrb.Spatial2DView(name="ext1",  origin="world/camera/ext1"),
                rrb.Spatial2DView(name="ext2",  origin="world/camera/ext2"),
                rrb.Spatial2DView(name="wrist", origin="world/camera/wrist"),
                rrb.TextDocumentView(name="task", origin="language"),
            ),
            column_shares=[3, 1],
        ),
        collapse_panels=True,
    )
    rr.init("droid_visualiser", spawn=False, default_blueprint=blueprint)

    # Robot base frame is Z-up right-handed — tell Rerun so the 3D view is oriented correctly
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # ── Camera intrinsics + static camera poses ─────────────────────────────
    # These make the viewer plot camera frustums in 3D, and allow DepthImage to render in 3D views.
    def log_camera_pinhole(camera_path: str, intrinsics: np.ndarray) -> None:
        # intrinsics: [fx, fy, cx, cy] at 180×320
        fx, fy, cx, cy = intrinsics
        rr.log(
            camera_path,
            rr.Pinhole(
                resolution=[float(_DEPTH_W), float(_DEPTH_H)],
                focal_length=[float(fx), float(fy)],
                principal_point=[float(cx), float(cy)],
            ),
        )

    if depth_intr_ext1 is not None:
        log_camera_pinhole("world/camera/ext1", depth_intr_ext1)
        # ext1_cam_extrinsics is static, so log once.
        rr.log(
            "world/camera/ext1",
            rr.Transform3D(
                mat3x3=T_ext1[:3, :3].astype(np.float32),
                translation=T_ext1[:3, 3].astype(np.float32),
                relation=rr.TransformRelation.ParentFromChild,
            ),
        )

    if depth_intr_wrist is not None:
        log_camera_pinhole("world/camera/wrist", depth_intr_wrist)
        # Wrist pose is time-varying (logged per-frame below).

    if depth_intr_ext2 is not None:
        log_camera_pinhole("world/camera/ext2", depth_intr_ext2)
        # Use static ext2 pose if available; otherwise keep it in camera-local coordinates.
        if T_ext2 is None:
            rr.log(
                "world/camera/ext2",
                rr.Transform3D(
                    mat3x3=np.eye(3, dtype=np.float32),
                    translation=np.zeros(3, dtype=np.float32),
                    relation=rr.TransformRelation.ParentFromChild,
                ),
            )
        else:
            rr.log(
                "world/camera/ext2",
                rr.Transform3D(
                    mat3x3=T_ext2[:3, :3].astype(np.float32),
                    translation=T_ext2[:3, 3].astype(np.float32),
                    relation=rr.TransformRelation.ParentFromChild,
                ),
            )

    # ── Log per-frame data ────────────────────────────────────────────────────
    for i, frame_idx in enumerate(frame_indices):
        rr.set_time("frame", sequence=int(frame_idx))

        merged_pts_parts: list[np.ndarray] = []
        merged_cols_parts: list[np.ndarray] = []

        # Language annotation (log at every frame so it shows on the timeline)
        rr.log("language", rr.TextDocument(task_text))

        # EEF position — 3D marker + projected into each camera as 2D overlay
        if eef_positions is not None:
            fi_eef = min(int(frame_idx), len(eef_positions) - 1)
            eef_xyz = eef_positions[fi_eef]
            rr.log("world/eef", rr.Points3D(
                [eef_xyz.tolist()],
                radii=0.02,
                colors=[[255, 0, 0]],
            ))

            # Project EEF into each static external camera (sanity check for extrinsics)
            eef_hom = np.array([eef_xyz[0], eef_xyz[1], eef_xyz[2], 1.0])
            for cam_path, T_cam2base, intr in [
                ("world/camera/ext1", T_ext1, depth_intr_ext1),
                ("world/camera/ext2", T_ext2, depth_intr_ext2),
            ]:
                if T_cam2base is None or intr is None:
                    continue
                T_base2cam = np.linalg.inv(T_cam2base)
                eef_cam = T_base2cam @ eef_hom          # (4,) in camera frame
                if eef_cam[2] <= 0:                     # behind camera
                    continue
                fx, fy, cx, cy = intr
                u = fx * eef_cam[0] / eef_cam[2] + cx
                v = fy * eef_cam[1] / eef_cam[2] + cy
                if 0 <= u < _DEPTH_W and 0 <= v < _DEPTH_H:
                    rr.log(f"{cam_path}/eef_projection", rr.Points2D(
                        [[float(u), float(v)]],
                        radii=5.0,
                        colors=[[255, 0, 0]],
                    ))

        # ── Per-camera: image + point cloud ───────────────────────────────────
        # ext1
        img_ext1 = cam_frames.get("exterior_image_1_left", {}).get(int(frame_idx))
        if depth_ext1 is None or depth_intr_ext1 is None:
            warnings.warn(f"ext1 depth missing at frame {frame_idx} — skipping")
        else:
            fi = min(int(frame_idx), depth_ext1.shape[0] - 1)
            if img_ext1 is not None:
                rr.log("world/camera/ext1/image", rr.Image(img_ext1))
            depth_img = depth_ext1[fi].astype(np.float32)
            depth_img = cv2.resize(depth_img, (_DEPTH_W, _DEPTH_H), interpolation=cv2.INTER_NEAREST)
            depth_img[~np.isfinite(depth_img) | (depth_img <= 0)] = 0.0
            rr.log("world/camera/ext1/depth", rr.DepthImage(depth_img, meter=1.0))
            if img_ext1 is not None:
                pts, cols = unproject_depth_frame(depth_ext1[fi], depth_intr_ext1, T_ext1, img_ext1)
                if len(pts) > 0:
                    rr.log("world/points/ext1", rr.Points3D(pts, colors=cols))
                    merged_pts_parts.append(pts)
                    merged_cols_parts.append(cols)

        # wrist
        img_wrist = cam_frames.get("wrist_image_left", {}).get(int(frame_idx))
        if depth_wrist is None or depth_intr_wrist is None:
            warnings.warn(f"wrist depth missing at frame {frame_idx} — skipping")
        else:
            fi = min(int(frame_idx), depth_wrist.shape[0] - 1)
            if wrist_dofs is not None:
                fi_dof = min(int(frame_idx), len(wrist_dofs) - 1)
                T_wrist = dof_to_mat(wrist_dofs[fi_dof])
            else:
                T_wrist = np.eye(4, dtype=np.float64)
            rr.log("world/camera/wrist", rr.Transform3D(
                mat3x3=T_wrist[:3, :3].astype(np.float32),
                translation=T_wrist[:3, 3].astype(np.float32),
                relation=rr.TransformRelation.ParentFromChild,
            ))
            if img_wrist is not None:
                rr.log("world/camera/wrist/image", rr.Image(img_wrist))
            depth_img = depth_wrist[fi].astype(np.float32)
            depth_img = cv2.resize(depth_img, (_DEPTH_W, _DEPTH_H), interpolation=cv2.INTER_NEAREST)
            depth_img[~np.isfinite(depth_img) | (depth_img <= 0)] = 0.0
            rr.log("world/camera/wrist/depth", rr.DepthImage(depth_img, meter=1.0))
            if img_wrist is not None:
                pts, cols = unproject_depth_frame(depth_wrist[fi], depth_intr_wrist, T_wrist, img_wrist)
                if len(pts) > 0:
                    rr.log("world/points/wrist", rr.Points3D(pts, colors=cols))
                    merged_pts_parts.append(pts)
                    merged_cols_parts.append(cols)

        # ext2
        img_ext2 = cam_frames.get("exterior_image_2_left", {}).get(int(frame_idx))
        if depth_ext2 is None or depth_intr_ext2 is None:
            warnings.warn(f"ext2 depth missing at frame {frame_idx} — skipping")
        elif T_ext2 is None:
            warnings.warn(f"ext2 extrinsics missing at frame {frame_idx} — skipping")
        else:
            fi = min(int(frame_idx), depth_ext2.shape[0] - 1)
            if img_ext2 is not None:
                rr.log("world/camera/ext2/image", rr.Image(img_ext2))
            depth_img = depth_ext2[fi].astype(np.float32)
            depth_img = cv2.resize(depth_img, (_DEPTH_W, _DEPTH_H), interpolation=cv2.INTER_NEAREST)
            depth_img[~np.isfinite(depth_img) | (depth_img <= 0)] = 0.0
            rr.log("world/camera/ext2/depth", rr.DepthImage(depth_img, meter=1.0))
            if img_ext2 is not None:
                pts, cols = unproject_depth_frame(depth_ext2[fi], depth_intr_ext2, T_ext2, img_ext2)
                if len(pts) > 0:
                    rr.log("world/points/ext2", rr.Points3D(pts, colors=cols))
                    merged_pts_parts.append(pts)
                    merged_cols_parts.append(cols)

        # ── Merged point cloud across cameras ───────────────────────────────
        if merged_pts_parts:
            merged_pts = np.concatenate(merged_pts_parts, axis=0)
            merged_cols = np.concatenate(merged_cols_parts, axis=0)
            rr.log("world/points/merged", rr.Points3D(merged_pts, colors=merged_cols))

    print("Rerun logging complete.")

    # ── Output ────────────────────────────────────────────────────────────────
    if args.serve:
        print(f"Serving gRPC at :{args.grpc_port}, web viewer at http://localhost:{args.web_port}")
        server_uri = rr.serve_grpc(grpc_port=args.grpc_port)
        rr.serve_web_viewer(connect_to=server_uri, web_port=args.web_port, open_browser=False)
        # Block until interrupted
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        safe_id = canonical_id.replace("/", "_").replace("+", "_")
        out_dir = Path("rerun")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"{safe_id}.rrd")
        rr.save(out_path)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
