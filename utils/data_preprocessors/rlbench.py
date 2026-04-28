import json
import math
from pathlib import Path

from kornia import augmentation as K
import numpy as np
import torch
from torch.nn import functional as F

from .base import DataPreprocessor


def _load_miscalibration_noise(level):
    """Load precomputed extrinsics noise from instructions/miscalibration_noise.json.

    Returns:
        cameras: list of camera name keys (ordered, matches cam_idx convention)
        noise:   dict {cam_name: {"R_noise": FloatTensor(3,3), "t_noise": FloatTensor(3)}}
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
        angle = float(np.linalg.norm(aa))
        if angle < 1e-12:
            R = np.eye(3, dtype=np.float64)
        else:
            axis = aa / angle
            K_skew = np.array([
                [ 0,        -axis[2],  axis[1]],
                [ axis[2],   0,       -axis[0]],
                [-axis[1],   axis[0],  0      ],
            ], dtype=np.float64)
            R = np.eye(3) + np.sin(angle) * K_skew + (1 - np.cos(angle)) * (K_skew @ K_skew)
        t = np.array(entry["translation_m"], dtype=np.float64)
        noise[cam_name] = {
            "R_noise": torch.tensor(R, dtype=torch.float32),
            "t_noise": torch.tensor(t, dtype=torch.float32),
        }

    return cameras, noise


def _load_task_extrinsics_offsets():
    offsets_path = Path(__file__).resolve().parents[2] / "instructions/peract2/task_extrinsics_offsets.json"
    if not offsets_path.exists():
        return {}
    with open(offsets_path) as f:
        data = json.load(f)
    return {
        task: (np.array(v["R"], dtype=np.float64), np.array(v["t"], dtype=np.float64))
        for task, v in data.items()
    }


def _apply_offset_to_extrinsics(ext, R, t, device, dtype):
    """new_cam_to_world = offset @ ext, where offset is built from (R, t)."""
    offset = torch.eye(4, device=device, dtype=dtype)
    offset[:3, :3] = torch.tensor(R, device=device, dtype=dtype)
    offset[:3, 3] = torch.tensor(t, device=device, dtype=dtype)
    return offset @ ext


class RLBenchDataPreprocessor(DataPreprocessor):

    def __init__(self, keypose_only=False, num_history=1,
                 orig_imsize=256, custom_imsize=None, depth2cloud=None,
                 rotate_pcd=False, rotate_angle_deg=0.0, rotate_axis='z',
                 use_front_camera_frame=False,
                 miscal_max_angle_deg=None, miscal_max_translation_m=None,
                 **kwargs):
        super().__init__(
            keypose_only=keypose_only,
            num_history=num_history,
            custom_imsize=custom_imsize,
            depth2cloud=depth2cloud
        )
        self.rotate_pcd = rotate_pcd
        self.rotate_angle_deg = rotate_angle_deg
        self.rotate_axis = rotate_axis
        self.miscal_max_angle_deg = miscal_max_angle_deg or 0.0
        self.miscal_max_translation_m = miscal_max_translation_m or 0.0
        self.aug = K.AugmentationSequential(
            K.RandomAffine(
                degrees=0,
                translate=0.0,
                scale=(0.75, 1.25),
                padding_mode="reflection",
                p=0.8
            ),
            K.RandomResizedCrop(
                size=(orig_imsize, orig_imsize),
                scale=(0.95, 1.05),
                p=0.1
            )
        ).cuda()

    def _sample_random_miscalibration(self, B, ncam, device, dtype):
        """Sample one random noise extrinsics perturbation per (B, ncam).

        Returns (B, ncam, 4, 4) transforms to left-multiply onto extrinsics.
        Sampled once per batch item so all nhist snapshots get the same noise.
        """
        # Random rotation via axis-angle: axis uniform on S², angle uniform in [-max, +max]
        axes = torch.randn(B, ncam, 3, device=device)
        axes = axes / (axes.norm(dim=-1, keepdim=True) + 1e-8)
        max_rad = self.miscal_max_angle_deg * math.pi / 180.0
        angles = (torch.rand(B, ncam, device=device) * 2 - 1) * max_rad  # (B, ncam)

        # Rodrigues: R = I + sin(θ)K + (1-cos(θ))K²
        kx, ky, kz = axes[..., 0], axes[..., 1], axes[..., 2]
        zeros = torch.zeros(B, ncam, device=device)
        K_skew = torch.stack([
            torch.stack([ zeros,   -kz,    ky], dim=-1),
            torch.stack([    kz, zeros,   -kx], dim=-1),
            torch.stack([   -ky,    kx, zeros], dim=-1),
        ], dim=-2)  # (B, ncam, 3, 3)
        I = torch.eye(3, device=device).expand(B, ncam, 3, 3)
        sin_a = angles.sin()[..., None, None]
        cos_a = angles.cos()[..., None, None]
        R = I + sin_a * K_skew + (1 - cos_a) * (K_skew @ K_skew)  # (B, ncam, 3, 3)

        # Random translation: uniform in [-max, +max] per axis
        t = (torch.rand(B, ncam, 3, device=device) * 2 - 1) * self.miscal_max_translation_m

        # Assemble 4×4
        T = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).expand(B, ncam, 4, 4).clone()
        T[..., :3, :3] = R.to(dtype)
        T[..., :3,  3] = t.to(dtype)
        return T

    def _rotate_point_cloud(self, pcd):
        """
        pcd: (B, ncam, 3, H, W)
        """

        angle = torch.tensor(self.rotate_angle_deg * math.pi / 180.0, device=pcd.device)

        c = torch.cos(angle)
        s = torch.sin(angle)

        if self.rotate_axis == 'z':
            R = torch.tensor([[c, -s, 0],
                              [s,  c, 0],
                              [0,  0, 1]], device=pcd.device)
        elif self.rotate_axis == 'y':
            R = torch.tensor([[ c, 0, s],
                              [ 0, 1, 0],
                              [-s, 0, c]], device=pcd.device)
        else:  # x
            R = torch.tensor([[1,  0,  0],
                              [0,  c, -s],
                              [0,  s,  c]], device=pcd.device)

        B, ncam, _, H, W = pcd.shape
        pcd_flat = pcd.reshape(B * ncam, 3, H * W)

        pcd_rot = torch.matmul(R, pcd_flat)
        return pcd_rot.reshape(B, ncam, 3, H, W)

    def process_obs(self, rgbs, rgb2d, depth, extrinsics, intrinsics,
                    augment=False, **kwargs):
        """
        RGBs of shape (B, ncam, 3, h_i, w_i) or (B, nhist, ncam, 3, h_i, w_i).
        depths of shape (B, ncam, h_i, w_i) or (B, nhist, ncam, h_i, w_i).
        extrinsics/intrinsics: (B, ncam, 4, 4)/(B, 3, 3) or (B, nhist, ncam, 4, 4)/(B, nhist, ncam, 3, 3).
        """
        has_hist = rgbs.ndim == 6
        if has_hist:
            B, nhist, ncam, C, H, W = rgbs.shape
            # Sample noise once per (B, ncam) and broadcast across nhist so all
            # history snapshots get identical miscalibration (it's a camera property).
            if self.miscal_max_angle_deg > 0 or self.miscal_max_translation_m > 0:
                noise_T = self._sample_random_miscalibration(B, ncam, extrinsics.device, extrinsics.dtype)
                extrinsics = noise_T.unsqueeze(1) @ extrinsics  # (B, 1, ncam, 4, 4) @ (B, nhist, ncam, 4, 4)
            rgbs = rgbs.view(B * nhist, ncam, C, H, W)
            depth = depth.view(B * nhist, ncam, *depth.shape[-2:])
            extrinsics = extrinsics.view(B * nhist, ncam, 4, 4)
            intrinsics = intrinsics.view(B * nhist, ncam, 3, 3)

        # Get point cloud from depth
        pcds = self.depth2cloud(
            depth.cuda(non_blocking=True).to(torch.bfloat16),
            extrinsics.cuda(non_blocking=True).to(torch.bfloat16),
            intrinsics.cuda(non_blocking=True).to(torch.bfloat16)
        )

        # Handle non-wrist cameras, which may require augmentations
        if augment:
            b, nc, _, h, w = rgbs.shape
            # Augment in half precision
            obs = torch.cat((
                rgbs.cuda(non_blocking=True).half() / 255,
                pcds[:, :rgbs.size(1)].half()
            ), 2)  # (B, ncam, 6, H, W)
            obs = obs.reshape(-1, 6, h, w)
            obs = self.aug(obs)
            # Convert to full precision
            rgb_3d = obs[:, :3].reshape(b, nc, 3, h, w).float()
            pcd_3d = obs[:, 3:].reshape(b, nc, 3, h, w).float()
        else:
            # Simply convert to full precision
            rgb_3d = rgbs.cuda(non_blocking=True).float() / 255
            pcd_3d = pcds[:, :rgb_3d.size(1)].float()
        if self.custom_imsize is not None and self.custom_imsize != rgb_3d.size(-1):
            b, nc, _, _, _ = rgb_3d.shape
            rgb_3d = F.interpolate(
                rgb_3d.flatten(0, 1), (self.custom_imsize, self.custom_imsize),
                mode='bilinear', antialias=True
            ).reshape(b, nc, -1, self.custom_imsize, self.custom_imsize)

        # Handle wrist cameras, no augmentations
        rgb_2d = None
        if rgb2d is not None:
            rgb_2d = rgb2d.cuda(non_blocking=True).float() / 255
            if self.custom_imsize is not None and self.custom_imsize != rgb_2d.size(-1):
                b, nc, _, _, _ = rgb_2d.shape
                rgb_2d = F.interpolate(
                    rgb_2d.flatten(0, 1), (self.custom_imsize, self.custom_imsize),
                    mode='bilinear', antialias=True
                ).reshape(b, nc, -1, self.custom_imsize, self.custom_imsize)

        # Concatenate
        if rgb_2d is not None:
            rgbs = torch.cat((rgb_3d, rgb_2d), 1)
        else:
            rgbs = rgb_3d
        if pcd_3d.size(1) < pcds.size(1):
            pcds = torch.cat((pcd_3d, pcds[:, :pcd_3d.size(1)].float()))
        else:
            pcds = pcd_3d

        if self.rotate_pcd:
            pcds = self._rotate_point_cloud(pcds)

        if has_hist:
            rgbs = rgbs.view(B, nhist, *rgbs.shape[1:])
            pcds = pcds.view(B, nhist, *pcds.shape[1:])

        return rgbs, pcds
