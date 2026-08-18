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


class RLBenchDataPreprocessor(DataPreprocessor):

    def __init__(self, keypose_only=False, num_history=1,
                 orig_imsize=256, custom_imsize=None, depth2cloud=None,
                 rotate_pcd=False, rotate_angle_deg=0.0, rotate_axis='z',
                 miscal_max_angle_deg=None, miscal_max_translation_m=None,
                 miscal_fixed_angle_deg=None, miscal_fixed_translation_m=None,
                 orbital_miscal_noise_level=None,
                 orbital_miscal_noise_file=None,
                 orbital_miscal_noise_levels=None,
                 cotrain_miscal_group_ids=None,
                 cotrain_miscal_level=None,
                 cotrain_miscal_levels=None,
                 noise_curriculum=False,
                 noise_curriculum_warmup_frac=1.0,
                 **kwargs):
        super().__init__(
            keypose_only=keypose_only,
            num_history=num_history,
            custom_imsize=custom_imsize,
            depth2cloud=depth2cloud
        )
        active = sum([
            orbital_miscal_noise_level is not None,
            bool(orbital_miscal_noise_levels),
            bool(cotrain_miscal_group_ids),
        ])
        if active > 1:
            raise ValueError(
                "orbital_miscal_noise_level, orbital_miscal_noise_levels, and "
                "cotrain_miscal_group_ids are mutually exclusive; set only one."
            )
        self.rotate_pcd = rotate_pcd
        self.rotate_angle_deg = rotate_angle_deg
        self.rotate_axis = rotate_axis
        self.miscal_max_angle_deg = miscal_max_angle_deg or 0.0
        self.miscal_max_translation_m = miscal_max_translation_m or 0.0
        # Fixed-magnitude variant: when >0, override the "max" sampler so each
        # perturbation has *exactly* this rotation angle / translation length
        # (random direction). Useful for plotting metric vs. noise magnitude.
        self.miscal_fixed_angle_deg = miscal_fixed_angle_deg or 0.0
        self.miscal_fixed_translation_m = miscal_fixed_translation_m or 0.0
        self._orbital_miscal_noise_level = orbital_miscal_noise_level
        # Selects which fixed-base JSON the per-group table is read from. None =
        # the pinned training file. An alternative file with the same schema
        # (e.g. the held-out seed-3187 one) yields a never-seen fixed base.
        self._orbital_miscal_noise_file = orbital_miscal_noise_file
        self._orbital_miscal_noise_levels = list(orbital_miscal_noise_levels) if orbital_miscal_noise_levels else None
        self._cotrain_miscal_group_ids = set(int(g) for g in cotrain_miscal_group_ids) if cotrain_miscal_group_ids else None
        self._cotrain_miscal_level = cotrain_miscal_level
        self._cotrain_miscal_levels = list(cotrain_miscal_levels) if cotrain_miscal_levels else None
        # Random mode: cotrain_miscal_group_ids + miscal_max_angle_deg (no file-based level needed)
        self._cotrain_random_mode = (
            bool(self._cotrain_miscal_group_ids)
            and (miscal_max_angle_deg or 0.0) > 0
            and cotrain_miscal_level is None
            and not cotrain_miscal_levels
        )
        if self._cotrain_miscal_group_ids is not None and not self._cotrain_random_mode \
                and (cotrain_miscal_level is None) == (self._cotrain_miscal_levels is None):
            raise ValueError(
                "cotrain_miscal_group_ids requires exactly one of cotrain_miscal_level "
                "(single level) or cotrain_miscal_levels (list, sampled per-sample), "
                "or set miscal_max_angle_deg>0 for per-group random noise."
            )
        self._group_noise_table        = None  # (K_groups,            ncam, 4, 4) CPU float32, lazy-init
        self._group_level_noise_table  = None  # (K_group_levels,      ncam, 4, 4) CPU float32, lazy-init
        self._group_level_key_to_row   = None  # {"G1_small": int, ...}

        self._cotrain_group_noise_table = None  # (K_groups, ncam, 4, 4) for cotrain_miscal_level, lazy-init
        self._miscal_logged = False
        self.noise_curriculum = noise_curriculum
        self.noise_curriculum_warmup_frac = max(float(noise_curriculum_warmup_frac), 1e-6)
        self._noise_progress = 0.0  # updated by trainer; 0 = no noise, 1 = full noise
        if noise_curriculum:
            print(f"[miscal] noise curriculum ENABLED: linear ramp over {noise_curriculum_warmup_frac:.1%} of training", flush=True)
        if orbital_miscal_noise_level is not None:
            extra = ""
            if self.miscal_max_angle_deg > 0 or self.miscal_max_translation_m > 0:
                extra = f" + random top-up max_angle={miscal_max_angle_deg}deg, max_t={miscal_max_translation_m}m"
            print(f"[miscal] per-group FILE: level='{orbital_miscal_noise_level}'{extra}", flush=True)

        elif self._orbital_miscal_noise_levels:
            print(f"[miscal] per-group MULTI-LEVEL (random): levels={self._orbital_miscal_noise_levels}", flush=True)
        elif self._cotrain_miscal_group_ids is not None:
            if self._cotrain_random_mode:
                level_str = f"RANDOM max_angle={miscal_max_angle_deg}deg, max_t={miscal_max_translation_m}m"
            elif self._cotrain_miscal_levels is not None:
                level_str = f"levels={self._cotrain_miscal_levels} (sampled per-sample)"
            else:
                level_str = f"level='{cotrain_miscal_level}'"
            print(
                f"[miscal] co-train MIXED: groups {sorted(self._cotrain_miscal_group_ids)} get "
                f"{level_str}, all others clean",
                flush=True,
            )
        elif self.miscal_fixed_angle_deg > 0 or self.miscal_fixed_translation_m > 0:
            print(
                f"[miscal] fixed-magnitude ENABLED: "
                f"angle={self.miscal_fixed_angle_deg}deg, "
                f"translation={self.miscal_fixed_translation_m}m",
                flush=True,
            )
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

    def set_noise_progress(self, p: float):
        """Set curriculum progress (0.0 = start, 1.0 = full noise). Called by trainer each step."""
        self._noise_progress = float(p)

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
        loader = lambda: _load_orbital_group_noise(
            self._orbital_miscal_noise_level, noise_file=self._orbital_miscal_noise_file
        )
        self._group_noise_table, groups, _ = self._build_noise_table(loader, ncam)
        print(
            f"[miscal] loaded from file: level='{self._orbital_miscal_noise_level}', "
            f"file={self._orbital_miscal_noise_file or 'default'}, K={len(groups)}, ncam={ncam}",
            flush=True,
        )

    def _ensure_group_level_noise_table(self, ncam):
        """Lazily load (K_group_levels, ncam, 4, 4) table with keys like 'G1_small'."""
        if self._group_level_noise_table is not None and self._group_level_noise_table.shape[1] == ncam:
            return
        self._group_level_noise_table, keys, self._group_level_key_to_row = self._build_noise_table(_load_orbital_group_level_noise, ncam)
        print(f"[miscal] per-group-level loaded: K={len(keys)}, ncam={ncam}", flush=True)

    def _ensure_cotrain_group_noise_table(self, ncam):
        """Lazily load (K_groups, ncam, 4, 4) table for co-training mixed-miscal mode."""
        if self._cotrain_group_noise_table is not None and self._cotrain_group_noise_table.shape[1] == ncam:
            return
        loader = lambda: _load_orbital_group_noise(self._cotrain_miscal_level)
        self._cotrain_group_noise_table, groups, _ = self._build_noise_table(loader, ncam)
        print(f"[miscal] cotrain group noise loaded: level='{self._cotrain_miscal_level}', K={len(groups)}, ncam={ncam}", flush=True)


    def _lookup_group_level_noise(self, camera_group, levels, ncam, device, dtype):
        """Shared helper: randomly pick a level per sample, look up (B, ncam, 4, 4) from the group-level table."""
        self._ensure_group_level_noise_table(ncam)
        rand_levels = [levels[i] for i in torch.randint(0, len(levels), (len(camera_group),)).tolist()]
        keys = [f"G{int(g)}_{l}" for g, l in zip(camera_group.tolist(), rand_levels)]
        idx = torch.tensor([self._group_level_key_to_row[k] for k in keys], dtype=torch.long)
        return self._group_level_noise_table[idx].to(device=device, dtype=dtype)

    def _get_miscal_noise(self, B, ncam, device, dtype, camera_group=None, task=None):
        """Return (B, ncam, 4, 4) noise transform, or None if miscal is disabled."""
        # Flatten to (B,) — dataset yields camera_group as (1,) or (1,1); collation
        # via torch.cat produces (B,) or (B,1). Squeeze to ensure 1-D indexing.
        if camera_group is not None:
            camera_group = camera_group.reshape(B)
        if self._cotrain_miscal_group_ids is not None and camera_group is not None:
            if self._cotrain_miscal_levels is not None:
                T = self._lookup_group_level_noise(camera_group, self._cotrain_miscal_levels, ncam, device, dtype)
            elif self._cotrain_miscal_level is not None:
                self._ensure_cotrain_group_noise_table(ncam)
                T = self._cotrain_group_noise_table[camera_group.long() - 1].to(device=device, dtype=dtype)
            else:
                # Random mode: freshly-sampled noise per batch, masked to miscal groups only
                T = self._sample_random_miscalibration(B, ncam, device, dtype)
            # Samples whose group is not in the miscal set get identity (clean extrinsics).
            ids = torch.tensor(sorted(self._cotrain_miscal_group_ids), dtype=camera_group.dtype)
            in_miscal = torch.isin(camera_group, ids).to(device=device).view(B, 1, 1, 1)
            eye = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).expand(B, ncam, 4, 4)
            return torch.where(in_miscal, T, eye)
        if self._orbital_miscal_noise_level is not None and camera_group is not None:
            self._ensure_group_noise_table(ncam)
            T_base = self._group_noise_table[camera_group.long() - 1].to(device=device, dtype=dtype)
            if self.miscal_max_angle_deg > 0 or self.miscal_max_translation_m > 0:
                T_rand = self._sample_random_miscalibration(B, ncam, device, dtype)
                return T_rand @ T_base
            return T_base
        if self._orbital_miscal_noise_levels is not None and camera_group is not None:
            return self._lookup_group_level_noise(camera_group, self._orbital_miscal_noise_levels, ncam, device, dtype)

        if (self.miscal_max_angle_deg > 0 or self.miscal_max_translation_m > 0
                or self.miscal_fixed_angle_deg > 0 or self.miscal_fixed_translation_m > 0):
            return self._sample_random_miscalibration(B, ncam, device, dtype)
        return None

    def _sample_random_miscalibration(self, B, ncam, device, dtype):
        """Sample one random noise extrinsics perturbation per (B, ncam).

        Returns (B, ncam, 4, 4) transforms to left-multiply onto extrinsics.
        Sampled once per batch item so all nhist snapshots get the same noise.
        """
        # Curriculum scale: linearly ramp from 0 to 1 over warmup_frac of training
        curriculum_scale = (
            min(1.0, self._noise_progress / self.noise_curriculum_warmup_frac)
            if self.noise_curriculum else 1.0
        )

        # Random rotation via axis-angle: axis uniform on S². Angle is either
        # uniform in [-max, +max] (random "noise budget" mode) or exactly the
        # fixed magnitude (deterministic-magnitude mode for sweeps).
        axes = torch.randn(B, ncam, 3, device=device)
        axes = axes / (axes.norm(dim=-1, keepdim=True) + 1e-8)
        if self.miscal_fixed_angle_deg > 0:
            rad = self.miscal_fixed_angle_deg * math.pi / 180.0 * curriculum_scale
            angles = torch.full((B, ncam), rad, device=device)
        else:
            max_rad = self.miscal_max_angle_deg * math.pi / 180.0 * curriculum_scale
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
        if self.miscal_fixed_translation_m > 0:
            t_dir = torch.randn(B, ncam, 3, device=device)
            t_dir = t_dir / (t_dir.norm(dim=-1, keepdim=True) + 1e-8)
            t = t_dir * (self.miscal_fixed_translation_m * curriculum_scale)
        else:
            t = (torch.rand(B, ncam, 3, device=device) * 2 - 1) * (self.miscal_max_translation_m * curriculum_scale)

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
