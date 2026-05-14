"""Shared miscalibration logic used by both training (RLBenchDataPreprocessor)
and online evaluation (utils_with_*_rlbench.py).

The module owns:
  * Noise-file loaders: instructions/miscalibration_noise.json and
    instructions/orbital_miscalibration_noise.json.
  * A canonical SE(3) application: T_new = T_noise @ T_old.
  * A small `build_pcd_from_obs` helper for online-eval call sites that
    construct (depth, extrinsics, intrinsics) stacks from an RLBench Observation.

There is one canonical convention for how miscalibration perturbs extrinsics:
  R_new = R_noise @ R_old
  t_new = R_noise @ t_old + t_noise
This matches `noise_T @ extrinsics` where noise_T is the 4×4 SE(3) form of
(R_noise, t_noise). Training and orbital eval already use this; bimanual and
peract online eval previously used a translation-only variant — `apply_miscalibration`
unifies them on the canonical form.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from utils.depth2cloud.rlbench import RLBenchDepth2Cloud


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _axis_angle_to_R(aa):
    """Rodrigues' formula. aa: (3,) numpy axis-angle in radians."""
    angle = float(np.linalg.norm(aa))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = aa / angle
    K_skew = np.array([
        [ 0,        -axis[2],  axis[1]],
        [ axis[2],   0,       -axis[0]],
        [-axis[1],   axis[0],  0      ],
    ], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * K_skew + (1 - np.cos(angle)) * (K_skew @ K_skew)


def noise_RT_from_dict(noise_dict, cameras):
    """Return (ncam,3,3) R and (ncam,3) t noise tensors from a {cam_name: {R_noise, t_noise}} dict."""
    ncam = len(cameras)
    R = torch.eye(3).unsqueeze(0).expand(ncam, 3, 3).clone()
    t = torch.zeros(ncam, 3)
    for c, cam in enumerate(cameras):
        if cam in noise_dict:
            R[c] = noise_dict[cam]["R_noise"]
            t[c] = noise_dict[cam]["t_noise"]
    return R, t


def per_cam_noise_T(noise_dict, cameras, ncam, dtype=torch.float32):
    """Build (ncam, 4, 4) SE(3) noise transforms from {cam_name: {R_noise, t_noise}}.

    Cameras missing from the dict, or beyond what `cameras` provides, get identity.
    """
    T = torch.eye(4, dtype=dtype).unsqueeze(0).expand(ncam, 4, 4).clone()
    upto = min(ncam, len(cameras))
    for c in range(upto):
        cam = cameras[c]
        if cam not in noise_dict:
            continue
        T[c, :3, :3] = noise_dict[cam]["R_noise"].to(dtype)
        T[c, :3,  3] = noise_dict[cam]["t_noise"].to(dtype)
    return T


def apply_miscalibration(extrinsics, T_noise):
    """Canonical SE(3) composition: T_new = T_noise @ extrinsics.

    Shapes supported:
      extrinsics (..., ncam, 4, 4) + T_noise (ncam, 4, 4)           — broadcast over batch
      extrinsics (B, ncam, 4, 4)   + T_noise (B, ncam, 4, 4)        — per-batch transform
      extrinsics (B, nhist, ncam, 4, 4) + T_noise (B, ncam, 4, 4)   — broadcast over nhist
    """
    T_noise = T_noise.to(device=extrinsics.device, dtype=extrinsics.dtype)
    if extrinsics.dim() == 5 and T_noise.dim() == 4:
        T_noise = T_noise.unsqueeze(1)
    return T_noise @ extrinsics


# ---------------------------------------------------------------------------
# Noise-file loaders
# ---------------------------------------------------------------------------

def _parse_noise_entries(cameras, keys, level_data, noise_path, section):
    noise = {}
    for key in keys:
        if key not in level_data:
            raise ValueError(f"Key '{key}' missing from {section} in {noise_path}")
        key_data = {}
        for cam_name in cameras:
            entry = level_data[key].get(cam_name)
            if entry is None:
                continue
            aa = np.array(entry["axis_angle_rad"], dtype=np.float64)
            t  = np.array(entry["translation_m"],  dtype=np.float64)
            key_data[cam_name] = {
                "R_noise": torch.tensor(_axis_angle_to_R(aa), dtype=torch.float32),
                "t_noise": torch.tensor(t, dtype=torch.float32),
            }
        noise[key] = key_data
    return noise


def _load_orbital_group_noise(level):
    """Load per-group orbital miscal noise from instructions/orbital_miscalibration_noise.json.

    Returns (cameras, groups, noise) where:
      cameras: list[str], camera names in file order
      groups:  list[str], e.g. ["G1", ..., "G6"]
      noise:   {group_str: {cam_name: {"R_noise": Tensor(3,3), "t_noise": Tensor(3)}}}
    """
    noise_path = Path(__file__).resolve().parents[2] / "instructions/orbital_miscalibration_noise.json"
    with open(noise_path) as f:
        data = json.load(f)

    cameras = data["cameras"]
    groups  = data["groups"]
    if level not in data["levels"]:
        raise ValueError(f"Unknown orbital miscal level '{level}'. Available: {list(data['levels'].keys())}")

    noise = _parse_noise_entries(cameras, groups, data["levels"][level], noise_path, f"levels.{level}")
    return cameras, groups, noise


def _load_orbital_task_group_noise(level):
    """Load per-(task, group) orbital miscal noise.

    Returns (cameras, task_group_keys, noise) where task_group_keys are strings
    like "place_cups_G1" and noise[key] is {cam_name: {R_noise, t_noise}}.
    """
    noise_path = Path(__file__).resolve().parents[2] / "instructions/orbital_miscalibration_noise.json"
    with open(noise_path) as f:
        data = json.load(f)

    if "per_task_group_levels" not in data:
        raise ValueError(
            f"No 'per_task_group_levels' section found in {noise_path}. "
            "Re-run scripts/generate_orbital_miscal_noise.py --overwrite to add it."
        )
    cameras         = data["cameras"]
    task_group_keys = data["task_group_keys"]
    if level not in data["per_task_group_levels"]:
        raise ValueError(f"Unknown per-task-group level '{level}'. Available: {list(data['per_task_group_levels'].keys())}")

    noise = _parse_noise_entries(cameras, task_group_keys, data["per_task_group_levels"][level], noise_path, f"per_task_group_levels.{level}")
    return cameras, task_group_keys, noise


def _load_orbital_group_level_noise():
    """Load per-(group, level) flat noise from the per_group_levels section.

    Returns (cameras, group_level_keys, noise) where group_level_keys are strings
    like "G1_small" and noise[key] is {cam_name: {R_noise, t_noise}}.
    """
    noise_path = Path(__file__).resolve().parents[2] / "instructions/orbital_miscalibration_noise.json"
    with open(noise_path) as f:
        data = json.load(f)

    if "per_group_levels" not in data:
        raise ValueError(
            f"No 'per_group_levels' section found in {noise_path}. "
            "Re-run scripts/generate_orbital_miscal_noise.py --overwrite to add it."
        )
    cameras          = data["cameras"]
    group_level_keys = data["group_level_keys"]
    noise = _parse_noise_entries(cameras, group_level_keys, data["per_group_levels"], noise_path, "per_group_levels")
    return cameras, group_level_keys, noise


def _load_miscalibration_noise(level):
    """Load precomputed extrinsics noise from instructions/miscalibration_noise.json.

    Returns (cameras, noise) where noise is {cam_name: {"R_noise": Tensor(3,3), "t_noise": Tensor(3)}}.
    """
    noise_path = Path(__file__).resolve().parents[2] / "instructions/miscalibration_noise.json"
    with open(noise_path) as f:
        data = json.load(f)

    cameras = data["cameras"]
    if level not in data["levels"]:
        raise ValueError(f"Unknown miscalibration level '{level}'. Available: {list(data['levels'].keys())}")

    level_data = data["levels"][level]
    noise = {}
    for cam_name in cameras:
        if cam_name not in level_data:
            continue
        entry = level_data[cam_name]
        aa = np.array(entry["axis_angle_rad"], dtype=np.float64)
        t  = np.array(entry["translation_m"], dtype=np.float64)
        noise[cam_name] = {
            "R_noise": torch.tensor(_axis_angle_to_R(aa), dtype=torch.float32),
            "t_noise": torch.tensor(t, dtype=torch.float32),
        }

    return cameras, noise


# ---------------------------------------------------------------------------
# Online-eval init/depth→PCD helpers
# ---------------------------------------------------------------------------

@dataclass
class MiscalibrationContext:
    """Bundle of state returned by `setup_miscalibration`. Fields are None when
    the corresponding feature is not configured."""
    cameras: list | None = None
    per_cam_noise: dict | None = None
    per_task_group_noise: dict | None = None
    depth2cloud: RLBenchDepth2Cloud | None = None


def setup_miscalibration(
    level=None,
    level_per_task_group=None,
    image_size=None,
    build_depth2cloud=True,
    log_prefix="[miscal]",
):
    """One-shot init for online-eval miscalibration setup.

    Args:
        level: name of a per-camera noise level in instructions/miscalibration_noise.json,
            or None to disable per-camera noise.
        level_per_task_group: name of a per-(task, group) level in
            instructions/orbital_miscalibration_noise.json (orbital eval only).
            Mutually exclusive with `level`.
        image_size: (h, w) or int; required when build_depth2cloud=True.
        build_depth2cloud: if True, construct an RLBenchDepth2Cloud module for
            the given image_size and return it on the context.
        log_prefix: prefix used in [miscal] log lines.

    Returns:
        MiscalibrationContext with whichever of (cameras, per_cam_noise,
        per_task_group_noise, depth2cloud) are configured.
    """
    if level is not None and level_per_task_group is not None:
        raise ValueError(
            "level and level_per_task_group are mutually exclusive; set only one."
        )

    cameras = None
    per_cam_noise = None
    per_task_group_noise = None

    if level is not None:
        cameras, per_cam_noise = _load_miscalibration_noise(level)
        print(
            f"{log_prefix} Miscalibration: level='{level}', cameras={cameras}",
            flush=True,
        )
    elif level_per_task_group is not None:
        cameras, _, per_task_group_noise = _load_orbital_task_group_noise(level_per_task_group)
        keys_preview = list(per_task_group_noise.keys())[:4]
        print(
            f"{log_prefix} Per-task-group miscalibration: level='{level_per_task_group}', "
            f"cameras={cameras}, keys={keys_preview}...",
            flush=True,
        )

    depth2cloud = None
    if build_depth2cloud:
        if image_size is None:
            raise ValueError("image_size is required when build_depth2cloud=True")
        h, w = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        depth2cloud = RLBenchDepth2Cloud((h, w))
        print(f"{log_prefix} depth2cloud built for image_size=({h}, {w})", flush=True)

    return MiscalibrationContext(
        cameras=cameras,
        per_cam_noise=per_cam_noise,
        per_task_group_noise=per_task_group_noise,
        depth2cloud=depth2cloud,
    )


def build_pcd_from_obs(
    obs,
    cameras,
    depth2cloud,
    T_noise=None,
    depth_accessor=getattr,
):
    """Build (1, ncam, 3, H, W) point cloud from an RLBench Observation.

    Reads `{cam}_depth`, `{cam}_camera_near`/`_far`, `{cam}_camera_extrinsics`,
    `{cam}_camera_intrinsics` for each camera in `cameras`, scales depth from
    [0, 1] to metric range, optionally applies SE(3) miscalibration `T_noise`,
    and runs `depth2cloud`.

    Args:
        obs: RLBench Observation.
        cameras: ordered list of camera names.
        depth2cloud: RLBenchDepth2Cloud instance.
        T_noise: optional (ncam, 4, 4) SE(3) noise transform.
        depth_accessor: callable (obs, attr_name) -> ndarray. Default `getattr`.
            Bimanual eval passes a lambda that reads from `obs.perception_data`.

    Returns:
        (1, ncam, 3, H, W) float32 CPU tensor.
    """
    depths, exts, ints = [], [], []
    for cam in cameras:
        depth_raw = depth_accessor(obs, f"{cam}_depth")
        near = obs.misc.get(f"{cam}_camera_near", 0.1)
        far  = obs.misc.get(f"{cam}_camera_far", 4.0)
        depths.append(torch.tensor(near + depth_raw * (far - near), dtype=torch.float32))
        exts.append(torch.tensor(obs.misc.get(f"{cam}_camera_extrinsics", np.eye(4)), dtype=torch.float32))
        ints.append(torch.tensor(obs.misc.get(f"{cam}_camera_intrinsics", np.eye(3)), dtype=torch.float32))

    depth      = torch.stack(depths).unsqueeze(0)       # (1, ncam, H, W)
    extrinsics = torch.stack(exts).unsqueeze(0)         # (1, ncam, 4, 4)
    intrinsics = torch.stack(ints).unsqueeze(0)         # (1, ncam, 3, 3)

    if T_noise is not None:
        extrinsics = apply_miscalibration(extrinsics, T_noise)

    pcd = depth2cloud(
        depth.cuda(non_blocking=True).to(torch.bfloat16),
        extrinsics.cuda(non_blocking=True).to(torch.bfloat16),
        intrinsics.cuda(non_blocking=True).to(torch.bfloat16),
    ).float().cpu()  # (1, ncam, 3, H, W)
    return pcd
