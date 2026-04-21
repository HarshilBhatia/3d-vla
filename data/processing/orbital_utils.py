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
    DEPTH_SCALE,
    num2id,
    image_to_float_array,
    keypoint_discovery,
)

CAMERAS  = ["orbital_left", "orbital_right", "wrist"]
CAM_KEYS = [None,           None,            None]
NHAND    = 1


def load_rgb(ep_path, cam, frame_id):
    """Load a PNG frame as (H, W, 3) uint8."""
    path = os.path.join(ep_path, "{}_rgb".format(cam), "{}.png".format(num2id(frame_id)))
    return np.array(Image.open(path).convert("RGB"))


def load_depth_metres(ep_path, cam, frame_id, obs, cam_key):
    """
    Load depth as float32 metres.
    Orbital cameras: PNG encodes absolute metres at DEPTH_SCALE.
    Standard cameras (e.g. wrist): PNG encodes [0,1]; unpack with near/far from obs.misc.
    """
    path = os.path.join(ep_path, "{}_depth".format(cam), "{}.png".format(num2id(frame_id)))
    d_raw = image_to_float_array(Image.open(path), DEPTH_SCALE)
    if cam_key is not None:
        near = obs.misc.get("{}_camera_near".format(cam_key), 0.0)
        far  = obs.misc.get("{}_camera_far".format(cam_key),  4.0)
        return (near + d_raw * (far - near)).astype(np.float32)
    return d_raw.astype(np.float32)


def load_extrinsics_from_misc(obs, cam_key):
    """4×4 cam-to-world from obs.misc."""
    key = "{}_camera_extrinsics".format(cam_key)
    E   = obs.misc.get(key, None)
    return np.array(E, dtype=np.float32) if E is not None else np.eye(4, dtype=np.float32)


def load_intrinsics_from_misc(obs, cam_key):
    """3×3 intrinsic matrix from obs.misc."""
    key = "{}_camera_intrinsics".format(cam_key)
    K   = obs.misc.get(key, None)
    return np.array(K, dtype=np.float32) if K is not None else np.eye(3, dtype=np.float32)


def load_orbital_extrinsics(ep_path):
    """Load pre-saved orbital camera extrinsics; falls back to identity matrices."""
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


def get_group_id(ep_path, group_str):
    """Return integer group id (1-6) from camera_group.txt or group string."""
    txt = os.path.join(ep_path, "camera_group.txt")
    if os.path.exists(txt):
        with open(txt) as f:
            return int(f.read().strip()[1])  # "G3" → 3
    return int(group_str[1])


def process_episode(ep_path, task_id, group_str, zarr_file, im_size=256):
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

        depth = np.stack([
            load_depth_metres(ep_path, cam, k, obs, cam_key)
            for cam, cam_key in zip(CAMERAS, CAM_KEYS)
        ]).astype(np.float16)[np.newaxis]

        E_wrist = load_extrinsics_from_misc(obs, "wrist")
        K_wrist = load_intrinsics_from_misc(obs, "wrist")
        extr = np.stack([E_ol, E_or, E_wrist]).astype(np.float16)[np.newaxis]
        intr = np.stack([K_ol, K_or, K_wrist]).astype(np.float16)[np.newaxis]

        state    = _eef(obs)
        state_p  = _eef(demo[key_frames[max(0, idx - 1)]])
        state_pp = _eef(demo[key_frames[max(0, idx - 2)]])
        prop     = np.stack([state_pp, state_p, state]).reshape(3, NHAND, 8)[np.newaxis]
        action   = _eef(obs_next).reshape(1, NHAND, 8)[np.newaxis]

        prop_j = _joints(obs).reshape(1, NHAND, 8)[np.newaxis]
        act_j  = _joints(obs_next).reshape(1, NHAND, 8)[np.newaxis]

        zarr_file["rgb"].append(rgb)
        zarr_file["depth"].append(depth)
        zarr_file["extrinsics"].append(extr)
        zarr_file["intrinsics"].append(intr)
        zarr_file["proprioception"].append(prop)
        zarr_file["action"].append(action)
        zarr_file["proprioception_joints"].append(prop_j)
        zarr_file["action_joints"].append(act_j)
        zarr_file["task_id"].append(np.array([task_id],  dtype=np.uint8))
        zarr_file["variation"].append(np.array([0],       dtype=np.uint8))
        zarr_file["camera_group"].append(np.array([group_id], dtype=np.uint8))
        n_written += 1

    return n_written
