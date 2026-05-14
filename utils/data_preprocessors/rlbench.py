import math

from kornia import augmentation as K
import numpy as np
import torch
from torch.nn import functional as F

from .base import DataPreprocessor
from .miscalibration import (
    _load_orbital_group_noise,
    _load_orbital_group_level_noise,
    _load_orbital_task_group_noise,
    noise_RT_from_dict,
    apply_miscalibration,
)


class RLBenchDataPreprocessor(DataPreprocessor):

    def __init__(self, keypose_only=False, num_history=1,
                 orig_imsize=256, custom_imsize=None, depth2cloud=None,
                 rotate_pcd=False, rotate_angle_deg=0.0, rotate_axis='z',
                 miscal_max_angle_deg=None, miscal_max_translation_m=None,
                 orbital_miscal_noise_level=None,
                 orbital_miscal_noise_level_per_task_group=None,
                 orbital_miscal_noise_levels=None,
                 **kwargs):
        super().__init__(
            keypose_only=keypose_only,
            num_history=num_history,
            custom_imsize=custom_imsize,
            depth2cloud=depth2cloud
        )
        active = sum([
            orbital_miscal_noise_level is not None,
            orbital_miscal_noise_level_per_task_group is not None,
            bool(orbital_miscal_noise_levels),
        ])
        if active > 1:
            raise ValueError(
                "orbital_miscal_noise_level, orbital_miscal_noise_level_per_task_group, and "
                "orbital_miscal_noise_levels are mutually exclusive; set only one."
            )
        self.rotate_pcd = rotate_pcd
        self.rotate_angle_deg = rotate_angle_deg
        self.rotate_axis = rotate_axis
        self.miscal_max_angle_deg = miscal_max_angle_deg or 0.0
        self.miscal_max_translation_m = miscal_max_translation_m or 0.0
        self._orbital_miscal_noise_level = orbital_miscal_noise_level
        self._orbital_miscal_noise_level_per_task_group = orbital_miscal_noise_level_per_task_group
        self._orbital_miscal_noise_levels = list(orbital_miscal_noise_levels) if orbital_miscal_noise_levels else None
        self._group_noise_table        = None  # (K_groups,            ncam, 4, 4) CPU float32, lazy-init
        self._group_level_noise_table  = None  # (K_group_levels,      ncam, 4, 4) CPU float32, lazy-init
        self._group_level_key_to_row   = None  # {"G1_small": int, ...}
        self._task_group_noise_table   = None  # (K_tasks * K_groups,  ncam, 4, 4) CPU float32, lazy-init
        self._task_group_key_to_row    = None  # {"task_G1": int}
        self._miscal_logged = False
        if orbital_miscal_noise_level is not None:
            print(f"[miscal] per-group FILE: level='{orbital_miscal_noise_level}'", flush=True)
        elif orbital_miscal_noise_level_per_task_group is not None:
            print(f"[miscal] per-task-group FILE: level='{orbital_miscal_noise_level_per_task_group}'", flush=True)
        elif self._orbital_miscal_noise_levels:
            print(f"[miscal] per-group MULTI-LEVEL (random): levels={self._orbital_miscal_noise_levels}", flush=True)
        elif self.miscal_max_angle_deg > 0 or self.miscal_max_translation_m > 0:
            print(f"[miscal] random ENABLED: max_angle={self.miscal_max_angle_deg}deg, max_translation={self.miscal_max_translation_m}m", flush=True)
        else:
            print("[miscal] disabled", flush=True)
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

    def _ensure_group_noise_table(self, ncam):
        """Lazily load the (K, ncam, 4, 4) noise table from instructions/orbital_miscalibration_noise.json.

        Table is indexed by (camera_group - 1), so row k corresponds to group G(k+1).
        """
        if self._group_noise_table is not None and self._group_noise_table.shape[1] == ncam:
            return
        file_cameras, groups, noise = _load_orbital_group_noise(self._orbital_miscal_noise_level)
        K = len(groups)
        table = torch.eye(4).view(1, 1, 4, 4).expand(K, ncam, 4, 4).clone()
        for k, group in enumerate(groups):
            R, t = noise_RT_from_dict(noise[group], file_cameras[:ncam])
            table[k, :, :3, :3] = R
            table[k, :, :3,  3] = t
        self._group_noise_table = table
        print(f"[miscal] loaded from file: level='{self._orbital_miscal_noise_level}', K={K}, ncam={ncam}", flush=True)

    def _ensure_group_level_noise_table(self, ncam):
        """Lazily load (K_group_levels, ncam, 4, 4) noise table from per_group_levels JSON section.

        Keys are like 'G1_small', 'G1_medium', 'G1_large', 'G2_small', ...
        During forward, one level per batch sample is drawn uniformly at random from
        _orbital_miscal_noise_levels and the key G{group}_{level} is used for lookup.
        """
        if self._group_level_noise_table is not None and self._group_level_noise_table.shape[1] == ncam:
            return
        file_cameras, group_level_keys, noise = _load_orbital_group_level_noise()
        K = len(group_level_keys)
        table = torch.eye(4).view(1, 1, 4, 4).expand(K, ncam, 4, 4).clone()
        for k, key in enumerate(group_level_keys):
            R, t = noise_RT_from_dict(noise[key], file_cameras[:ncam])
            table[k, :, :3, :3] = R
            table[k, :, :3,  3] = t
        self._group_level_noise_table = table
        self._group_level_key_to_row  = {k: i for i, k in enumerate(group_level_keys)}
        print(f"[miscal] per-group-level loaded: K={K}, ncam={ncam}", flush=True)

    def _ensure_task_group_noise_table(self, ncam):
        """Lazily load the (K_task_groups, ncam, 4, 4) noise table for per-(task, group) miscal."""
        if self._task_group_noise_table is not None and self._task_group_noise_table.shape[1] == ncam:
            return
        file_cameras, task_group_keys, noise = _load_orbital_task_group_noise(self._orbital_miscal_noise_level_per_task_group)
        K = len(task_group_keys)
        table = torch.eye(4).view(1, 1, 4, 4).expand(K, ncam, 4, 4).clone()
        for k, key in enumerate(task_group_keys):
            R, t = noise_RT_from_dict(noise[key], file_cameras[:ncam])
            table[k, :, :3, :3] = R
            table[k, :, :3,  3] = t
        self._task_group_noise_table = table
        self._task_group_key_to_row = {k: i for i, k in enumerate(task_group_keys)}
        print(f"[miscal] per-task-group loaded: level='{self._orbital_miscal_noise_level_per_task_group}', K={K}, ncam={ncam}", flush=True)

    def _get_miscal_noise(self, B, ncam, device, dtype, camera_group=None, task=None):
        """Return (B, ncam, 4, 4) noise transform, or None if miscal is disabled."""
        if self._orbital_miscal_noise_level is not None and camera_group is not None:
            self._ensure_group_noise_table(ncam)
            idx = camera_group.long() - 1  # (B,) 0-based
            return self._group_noise_table[idx].to(device=device, dtype=dtype)  # (B, ncam, 4, 4)
        if self._orbital_miscal_noise_levels is not None and camera_group is not None:
            self._ensure_group_level_noise_table(ncam)
            levels = self._orbital_miscal_noise_levels
            rand_levels = [levels[i] for i in torch.randint(0, len(levels), (B,)).tolist()]
            keys = [f"G{int(g)}_{l}" for g, l in zip(camera_group.tolist(), rand_levels)]
            idx = torch.tensor([self._group_level_key_to_row[k] for k in keys], dtype=torch.long)
            return self._group_level_noise_table[idx].to(device=device, dtype=dtype)
        if self._orbital_miscal_noise_level_per_task_group is not None:
            if task is None or camera_group is None:
                return None
            self._ensure_task_group_noise_table(ncam)
            keys = [f"{t}_G{int(g)}" for t, g in zip(task, camera_group)]
            idx = torch.tensor(
                [self._task_group_key_to_row[k] for k in keys],
                dtype=torch.long,
            )
            return self._task_group_noise_table[idx].to(device=device, dtype=dtype)
        if self.miscal_max_angle_deg > 0 or self.miscal_max_translation_m > 0:
            return self._sample_random_miscalibration(B, ncam, device, dtype)
        return None

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
                    augment=False, camera_group=None, task=None, **kwargs):
        """
        RGBs of shape (B, ncam, 3, h_i, w_i) or (B, nhist, ncam, 3, h_i, w_i).
        depths of shape (B, ncam, h_i, w_i) or (B, nhist, ncam, h_i, w_i).
        extrinsics/intrinsics: (B, ncam, 4, 4)/(B, 3, 3) or (B, nhist, ncam, 4, 4)/(B, nhist, ncam, 3, 3).
        camera_group: (B,) uint8 tensor with group ids (1-based), or None.
        task: list of B task name strings, or None.
        """
        has_hist = rgbs.ndim == 6
        if has_hist:
            B, nhist, ncam, C, H, W = rgbs.shape
        else:
            B, ncam, C, H, W = rgbs.shape

        # Apply miscalibration noise once per (B, ncam); apply_miscalibration broadcasts over nhist.
        noise_T = self._get_miscal_noise(B, ncam, extrinsics.device, extrinsics.dtype, camera_group, task)
        if noise_T is not None:
            extrinsics = apply_miscalibration(extrinsics, noise_T)

        if has_hist:
            rgbs = rgbs.view(B * nhist, ncam, C, H, W)
            depth = depth.view(B * nhist, ncam, *depth.shape[-2:])
            extrinsics = extrinsics.view(B * nhist, ncam, 4, 4)
            intrinsics = intrinsics.view(B * nhist, ncam, 3, 3)

        # Get point cloud from depth
        pcds = self.depth2cloud(
            depth.to(device='cuda', dtype=torch.bfloat16, non_blocking=True),
            extrinsics.to(device='cuda', dtype=torch.bfloat16, non_blocking=True),
            intrinsics.to(device='cuda', dtype=torch.bfloat16, non_blocking=True),
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

        # if self.rotate_pcd:
        #     pcds = self._rotate_point_cloud(pcds)

        if has_hist:
            rgbs = rgbs.view(B, nhist, *rgbs.shape[1:])
            pcds = pcds.view(B, nhist, *pcds.shape[1:])

        return rgbs, pcds
