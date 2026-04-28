"""
Shared utilities for the orbital camera pipeline.

Used by both data/generation/orbital/to_zarr.py and data/processing/convert_to_zarr/orbital_to_zarr.py.
"""

import os
import pickle

import numpy as np
from PIL import Image

from data.processing.rlbench_utils import (
    CustomUnpickler,
    num2id,
    keypoint_discovery,
)

CAMERAS = ["orbital_left", "orbital_right", "wrist"]
NHAND   = 1


def load_rgb(ep_path, cam, frame_id):
    """Load a PNG frame as (H, W, 3) uint8."""
    path = os.path.join(ep_path, "{}_rgb".format(cam), "{}.png".format(num2id(frame_id)))
    return np.array(Image.open(path).convert("RGB"))



def load_extrinsics_from_misc(obs, cam_key):
    """4×4 cam-to-world from obs.misc."""
    key = "{}_camera_extrinsics".format(cam_key)
    E   = obs.misc.get(key)
    return np.array(E, dtype=np.float32)


def load_intrinsics_from_misc(obs, cam_key):
    """3×3 intrinsic matrix from obs.misc."""
    key = "{}_camera_intrinsics".format(cam_key)
    K   = obs.misc.get(key)
    return np.array(K, dtype=np.float32) 


def load_orbital_extrinsics(ep_path):
    """Load pre-saved orbital camera extrinsics/intrinsics.

    Returns (E_left, E_right, K_left, K_right).
    Falls back to identity matrices if the pkl is missing.
    """
    path = os.path.join(ep_path, "orbital_extrinsics.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return (
            np.array(data["left_extrinsics"],  dtype=np.float32),
            np.array(data["right_extrinsics"], dtype=np.float32),
            np.array(data["left_intrinsics"],  dtype=np.float32),
            np.array(data["right_intrinsics"], dtype=np.float32),
        )
    eye4, eye3 = np.eye(4, dtype=np.float32), np.eye(3, dtype=np.float32)
    return eye4, eye4, eye3, eye3


def depth_to_pcd_numpy(depth, extrinsics, intrinsics):
    """Unproject depth (H,W) float32 metres → world-space pcd (3,H,W) float32.

    Matches RLBenchDepth2Cloud exactly: builds [u*d, v*d, d, 1] then applies
    (K @ [R^T | -R^T@C])^{-1}.
    """
    H, W = depth.shape
    u = np.tile(np.arange(W, dtype=np.float32), (H, 1))
    v = np.tile(np.arange(H, dtype=np.float32), (W, 1)).T
    uv1d = np.stack(
        [u * depth, v * depth, depth, np.ones((H, W), dtype=np.float32)], axis=0
    )  # (4, H, W)
    R = extrinsics[:3, :3].astype(np.float32)
    C = extrinsics[:3, 3:].astype(np.float32)
    ext_inv      = np.concatenate([R.T, -R.T @ C], axis=1)        # (3, 4)
    cam_proj     = intrinsics.astype(np.float32) @ ext_inv         # (3, 4)
    cam_proj_4x4 = np.vstack([cam_proj, [[0, 0, 0, 1]]])           # (4, 4)
    cam_proj_inv = np.linalg.inv(cam_proj_4x4)[:3]                 # (3, 4)
    return (cam_proj_inv @ uv1d.reshape(4, -1)).reshape(3, H, W).astype(np.float32)


def get_group_id(ep_path, group_str):
    """Return integer group id (1-6) from camera_group.txt or group string."""
    txt = os.path.join(ep_path, "camera_group.txt")
    if os.path.exists(txt):
        with open(txt) as f:
            return int(f.read().strip()[1])  # "G3" → 3
    return int(group_str[1])


def process_episode(ep_path, task_id, group_str, zarr_file, im_size=256, demo_id=0):
    """
    Extract keyframes from one orbital episode and append rows to zarr_file.
    Returns number of keyframes written.
    """
    low_dim_path = os.path.join(ep_path, "low_dim_obs.pkl")
    if not os.path.exists(low_dim_path):
        print("[WARN] Missing low_dim_obs.pkl in {}".format(ep_path))
        return 0

    with open(low_dim_path, "rb") as f:
        demo = CustomUnpickler(f).load()

    key_frames = keypoint_discovery(demo, bimanual=False)
    key_frames.insert(0, 0)
    if len(key_frames) < 2:
        print("[WARN] Not enough keyframes in {}".format(ep_path))
        return 0

    E_ol, E_or, K_ol, K_or = load_orbital_extrinsics(ep_path)
    group_id = get_group_id(ep_path, group_str)

    def _eef(o):
        return np.concatenate([o.gripper_pose, [o.gripper_open]]).astype(np.float32)

    def _joints(o):
        return np.concatenate([o.joint_positions, [o.gripper_open]]).astype(np.float32)

    n_written = 0
    for idx, k in enumerate(key_frames[:-1]):
        obs      = demo[k]
        obs_next = demo[key_frames[idx + 1]]

        rgb = np.stack([
            load_rgb(ep_path, cam, k).transpose(2, 0, 1)
            for cam in CAMERAS
        ])[np.newaxis]

        # All depths live in the pkl obs — no PNG roundtrip.
        # Orbital: metres directly. Wrist: [0,1] normalized, convert with near/far from obs.misc.
        near_wr = obs.misc.get("wrist_camera_near")
        far_wr  = obs.misc.get("wrist_camera_far")
        depth_l = obs.orbital_left_depth.astype(np.float32)
        depth_r = obs.orbital_right_depth.astype(np.float32)
        depth_w = (near_wr + obs.wrist_depth * (far_wr - near_wr)).astype(np.float32)

        E_wrist = load_extrinsics_from_misc(obs, "wrist")
        K_wrist = load_intrinsics_from_misc(obs, "wrist")
        extr = np.stack([E_ol, E_or, E_wrist]).astype(np.float16)[np.newaxis]
        intr = np.stack([K_ol, K_or, K_wrist]).astype(np.float16)[np.newaxis]

        pcd = np.stack([
            depth_to_pcd_numpy(depth_l, E_ol,    K_ol),
            depth_to_pcd_numpy(depth_r, E_or,    K_or),
            depth_to_pcd_numpy(depth_w, E_wrist, K_wrist),
        ]).astype(np.float16)[np.newaxis]

        state    = _eef(obs)
        state_p  = _eef(demo[key_frames[max(0, idx - 1)]])
        state_pp = _eef(demo[key_frames[max(0, idx - 2)]])
        prop     = np.stack([state_pp, state_p, state]).reshape(3, NHAND, 8)[np.newaxis]
        action   = _eef(obs_next).reshape(1, NHAND, 8)[np.newaxis]

        prop_j = _joints(obs).reshape(1, NHAND, 8)[np.newaxis]
        act_j  = _joints(obs_next).reshape(1, NHAND, 8)[np.newaxis]

        zarr_file["rgb"].append(rgb)
        zarr_file["pcd"].append(pcd)
        zarr_file["extrinsics"].append(extr)
        zarr_file["intrinsics"].append(intr)
        zarr_file["proprioception"].append(prop)
        zarr_file["action"].append(action)
        zarr_file["proprioception_joints"].append(prop_j)
        zarr_file["action_joints"].append(act_j)
        zarr_file["task_id"].append(np.array([task_id],  dtype=np.uint8))
        zarr_file["variation"].append(np.array([0],       dtype=np.uint8))
        zarr_file["camera_group"].append(np.array([group_id], dtype=np.uint8))
        zarr_file["demo_id"].append(np.array([demo_id],  dtype=np.uint32))
        n_written += 1

    return n_written
