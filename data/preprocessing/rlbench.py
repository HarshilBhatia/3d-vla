import math

from kornia import augmentation as K
import numpy as np
import torch
from torch.nn import functional as F

from .base import DataPreprocessor
from .miscalibration import (
    _load_orbital_group_noise,
    _load_orbital_group_level_noise,

    per_cam_noise_T,
    apply_miscalibration,
)


# A PERSISTENT per-camera-group calibration error. Orthogonal to perturbation
# noise, which is per-sample jitter with no bias.
MISCAL_MODES = ("none", "group")


class RLBenchDataPreprocessor(DataPreprocessor):

    def __init__(self, keypose_only=False, visual_num_history=1, proprio_num_history=None,
                 orig_imsize=256, custom_imsize=None, depth2cloud=None,
                 rotate_pcd=False, rotate_angle_deg=0.0, rotate_axis='z',
                 miscal_mode='none',
                 perturbation_noise_rot_deg=None, perturbation_noise_trans_m=None,
                 perturbation_noise_fixed_rot_deg=None, perturbation_noise_fixed_trans_m=None,
                 miscal_group_level=None,
                 miscal_group_file=None,
                 miscal_camera_groups=None,
                 miscal_cameras=None,
                 **kwargs):
        super().__init__(
            keypose_only=keypose_only,
            visual_num_history=visual_num_history,
            proprio_num_history=proprio_num_history,
            custom_imsize=custom_imsize,
            depth2cloud=depth2cloud
        )
        if miscal_mode not in MISCAL_MODES:
            raise ValueError(f"miscal_mode must be one of {sorted(MISCAL_MODES)}, got {miscal_mode!r}")
        self.miscal_mode = miscal_mode
        self.rotate_pcd = rotate_pcd
        self.rotate_angle_deg = rotate_angle_deg
        self.rotate_axis = rotate_axis

        # Per-sample jitter, no persistent bias: extrinsics stay correct on
        # average. Composes on top of miscalibration when both are active.
        self.perturbation_noise_rot_deg = perturbation_noise_rot_deg or 0.0
        self.perturbation_noise_trans_m = perturbation_noise_trans_m or 0.0
        # Every draw is exactly this magnitude (random direction), for sweeps.
        self.perturbation_noise_fixed_rot_deg = perturbation_noise_fixed_rot_deg or 0.0
        self.perturbation_noise_fixed_trans_m = perturbation_noise_fixed_trans_m or 0.0
        self._has_perturbation = bool(
            self.perturbation_noise_rot_deg or self.perturbation_noise_trans_m
            or self.perturbation_noise_fixed_rot_deg or self.perturbation_noise_fixed_trans_m
        )

        # Scalar level -> levels[<level>]; list -> per_group_levels, one draw/sample.
        self._group_level = miscal_group_level
        self._group_levels = (
            list(miscal_group_level) if isinstance(miscal_group_level, (list, tuple)) else None
        )
        if self._group_levels is not None:
            self._group_level = None
        # None = the pinned training file; a same-schema alternative gives a
        # never-seen base at the same magnitude.
        self._group_file = miscal_group_file
        # None = every group; otherwise all other groups stay clean.
        self._camera_groups = (
            None if miscal_camera_groups is None
            else set(int(g) for g in miscal_camera_groups)
        )
        self._miscal_cameras = (
            None if miscal_cameras is None else tuple(sorted(set(int(i) for i in miscal_cameras)))
        )

        if miscal_mode == 'group' and miscal_group_level is None:
            raise ValueError("miscal_mode='group' requires miscal_group_level (a level name or a list of them)")
        if miscal_mode == 'none' and miscal_group_level is not None:
            raise ValueError("miscal_group_level is set but miscal_mode='none'; set miscal_mode='group' to use it")

        self._group_noise_table       = None  # (K_groups,       ncam, 4, 4) lazy-init
        self._group_level_noise_table = None  # (K_group_levels, ncam, 4, 4) lazy-init
        self._group_level_key_to_row  = None  # {"G1_small": int, ...}
        self._miscal_logged = False
        print(f"[miscal] {self._describe()}", flush=True)
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

    def _describe(self):
        """One line naming the active regimes, for the startup log."""
        parts = []
        if self.miscal_mode == 'group':
            level = self._group_levels if self._group_levels is not None else self._group_level
            parts.append(f"miscalibration: per-group fixed, level={level!r}, "
                         f"file={self._group_file or 'default'}")
        if self._has_perturbation:
            if self.perturbation_noise_fixed_rot_deg or self.perturbation_noise_fixed_trans_m:
                mag = (f"fixed rot={self.perturbation_noise_fixed_rot_deg}deg, "
                       f"trans={self.perturbation_noise_fixed_trans_m}m")
            else:
                mag = (f"rot=+-{self.perturbation_noise_rot_deg}deg, "
                       f"trans=+-{self.perturbation_noise_trans_m}m")
            parts.append(f"perturbation noise: {mag}")
        if not parts:
            return "disabled"
        desc = "; ".join(parts)
        if self._camera_groups is not None:
            desc += f"; only camera groups {sorted(self._camera_groups)} (others clean)"
        if self._miscal_cameras is not None:
            desc += f"; only cameras {list(self._miscal_cameras)}"
        return desc

    def _build_noise_table(self, loader_fn, ncam):
        """Build a (K, ncam, 4, 4) noise table from a loader function.

        Returns (table, keys, key_to_row) where key_to_row maps key string → row index.
        loader_fn() must return (file_cameras, keys, noise_dict).
        """
        file_cameras, keys, noise = loader_fn()
        K = len(keys)
        table = torch.eye(4).view(1, 1, 4, 4).expand(K, ncam, 4, 4).clone()
        for k, key in enumerate(keys):
            table[k] = per_cam_noise_T(noise[key], file_cameras[:ncam], ncam)
        key_to_row = {k: i for i, k in enumerate(keys)}
        return table, keys, key_to_row

    def _ensure_group_noise_table(self, ncam):
        """Lazily load (K, ncam, 4, 4) table indexed by (camera_group - 1)."""
        if self._group_noise_table is not None and self._group_noise_table.shape[1] == ncam:
            return
        loader = lambda: _load_orbital_group_noise(self._group_level, noise_file=self._group_file)
        self._group_noise_table, groups, _ = self._build_noise_table(loader, ncam)
        print(
            f"[miscal] loaded from file: level='{self._group_level}', "
            f"file={self._group_file or 'default'}, K={len(groups)}, ncam={ncam}",
            flush=True,
        )

    def _ensure_group_level_noise_table(self, ncam):
        """Lazily load (K_group_levels, ncam, 4, 4) table with keys like 'G1_small'."""
        if self._group_level_noise_table is not None and self._group_level_noise_table.shape[1] == ncam:
            return
        self._group_level_noise_table, keys, self._group_level_key_to_row = \
            self._build_noise_table(_load_orbital_group_level_noise, ncam)
        print(f"[miscal] per-group-level loaded: K={len(keys)}, ncam={ncam}", flush=True)

    def _lookup_group_level_noise(self, camera_group, levels, ncam, device, dtype):
        """Shared helper: randomly pick a level per sample, look up (B, ncam, 4, 4) from the group-level table."""
        self._ensure_group_level_noise_table(ncam)
        rand_levels = [levels[i] for i in torch.randint(0, len(levels), (len(camera_group),)).tolist()]
        keys = [f"G{int(g)}_{l}" for g, l in zip(camera_group.tolist(), rand_levels)]
        idx = torch.tensor([self._group_level_key_to_row[k] for k in keys], dtype=torch.long)
        return self._group_level_noise_table[idx].to(device=device, dtype=dtype)

    def _get_miscal_noise(self, B, ncam, device, dtype, camera_group=None, task=None):
        """Return (B, ncam, 4, 4) noise transform, or None if miscal is disabled."""
        if self.miscal_mode == 'none' and not self._has_perturbation:
            return None
        # Flatten to (B,) -- dataset yields camera_group as (1,) or (1,1); collation
        # via torch.cat produces (B,) or (B,1). Squeeze to ensure 1-D indexing.
        if camera_group is not None:
            camera_group = camera_group.reshape(B)

        T = None
        if self.miscal_mode == 'group':
            if camera_group is None:
                raise ValueError("miscal_mode='group' needs camera_group; the zarr has none")
            if self._group_levels is not None:
                T = self._lookup_group_level_noise(camera_group, self._group_levels, ncam, device, dtype)
            else:
                self._ensure_group_noise_table(ncam)
                T = self._group_noise_table[camera_group.long() - 1].to(device=device, dtype=dtype)

        if self._has_perturbation:
            P = self._sample_random_perturbation(B, ncam, device, dtype)
            T = P if T is None else P @ T

        return self._mask_miscal_cameras(self._mask_camera_groups(T, camera_group, B, ncam, device, dtype), ncam)

    def _mask_camera_groups(self, transforms, camera_group, B, ncam, device, dtype):
        """Samples whose camera group is not selected keep clean extrinsics."""
        if self._camera_groups is None:
            return transforms
        if camera_group is None:
            raise ValueError("miscal_camera_groups is set but the zarr has no camera_group")
        ids = torch.tensor(sorted(self._camera_groups), dtype=camera_group.dtype)
        selected = torch.isin(camera_group, ids).to(device=device).view(B, 1, 1, 1)
        eye = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).expand(B, ncam, 4, 4)
        return torch.where(selected, transforms, eye)

    def _mask_miscal_cameras(self, transforms, ncam):
        """Keep non-selected cameras geometrically clean (identity transform)."""
        if self._miscal_cameras is None:
            return transforms
        invalid = [i for i in self._miscal_cameras if i >= ncam]
        if invalid:
            raise ValueError(f"miscal_cameras={invalid} outside ncam={ncam}")
        enabled = torch.zeros(ncam, dtype=torch.bool, device=transforms.device)
        enabled[list(self._miscal_cameras)] = True
        eye = torch.eye(4, dtype=transforms.dtype, device=transforms.device).view(1, 1, 4, 4)
        return torch.where(enabled.view(1, ncam, 1, 1), transforms, eye)

    def _sample_random_perturbation(self, B, ncam, device, dtype):
        """Sample one random extrinsics perturbation per (B, ncam).

        Returns (B, ncam, 4, 4) transforms to left-multiply onto extrinsics.
        Drawn once per batch item so all nhist snapshots share the same jitter.
        """
        max_rot_deg = self.perturbation_noise_rot_deg
        max_trans_m = self.perturbation_noise_trans_m
        # Random rotation via axis-angle: axis uniform on S². Angle is either
        # uniform in [-max, +max] (random "noise budget" mode) or exactly the
        # fixed magnitude (deterministic-magnitude mode for sweeps).
        axes = torch.randn(B, ncam, 3, device=device)
        axes = axes / (axes.norm(dim=-1, keepdim=True) + 1e-8)
        if self.perturbation_noise_fixed_rot_deg > 0:
            rad = self.perturbation_noise_fixed_rot_deg * math.pi / 180.0
            angles = torch.full((B, ncam), rad, device=device)
        else:
            max_rad = max_rot_deg * math.pi / 180.0
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

        # Random translation: either uniform-in-cube up to ±max per axis, or a
        # uniform unit direction times a fixed length (sweep mode).
        if self.perturbation_noise_fixed_trans_m > 0:
            t_dir = torch.randn(B, ncam, 3, device=device)
            t_dir = t_dir / (t_dir.norm(dim=-1, keepdim=True) + 1e-8)
            t = t_dir * self.perturbation_noise_fixed_trans_m
        else:
            t = (torch.rand(B, ncam, 3, device=device) * 2 - 1) * max_trans_m

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
                rgbs.to(device='cuda', dtype=torch.float16, non_blocking=True) / 255,
                pcds[:, :rgbs.size(1)].half()
            ), 2)  # (B, ncam, 6, H, W)
            obs = obs.reshape(-1, 6, h, w)
            obs = self.aug(obs)
            # Convert to full precision
            rgb_3d = obs[:, :3].reshape(b, nc, 3, h, w).float()
            pcd_3d = obs[:, 3:].reshape(b, nc, 3, h, w).float()
        else:
            # Simply convert to full precision
            rgb_3d = rgbs.to(device='cuda', dtype=torch.float32, non_blocking=True) / 255
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
            rgb_2d = rgb2d.to(device='cuda', dtype=torch.float32, non_blocking=True) / 255
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
