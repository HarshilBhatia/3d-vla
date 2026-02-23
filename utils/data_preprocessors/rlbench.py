from kornia import augmentation as K
import json
import os
import time
import numpy as np
import torch
from torch.nn import functional as F

from .base import DataPreprocessor



def _load_task_extrinsics_offsets(path=None):
    """Load per-task R (3x3) and t (3) from JSON. Returns dict task_name -> (R, t) as numpy."""
    
    path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "instructions", "peract2", "task_extrinsics_offsets.json"
    )
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for task, data in raw.items():
        R = np.array(data["R"], dtype=np.float64)
        t = np.array(data["t"], dtype=np.float64)
        out[task] = (R, t)
    return out


def _apply_offset_to_extrinsics(extrinsics_i0, R, t, device, dtype):
    """Apply world-frame offset to front cam: new_cam_to_world = [R|t;0 0 0 1] @ cam_to_world.
    extrinsics_i0: (4, 4) camera-to-world for one sample.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    T = torch.tensor(T, device=device, dtype=dtype)
    return T @ extrinsics_i0


class RLBenchDataPreprocessor(DataPreprocessor):

    def __init__(self, keypose_only=False, num_history=1,
                 orig_imsize=256, custom_imsize=None, depth2cloud=None,
                 use_front_camera_frame=False, pc_rotate_by_front_camera=False,
                 task_extrinsics_offsets_path=None):
        super().__init__(
            keypose_only=keypose_only,
            num_history=num_history,
            custom_imsize=custom_imsize,
            depth2cloud=depth2cloud,
            use_front_camera_frame=use_front_camera_frame,
            pc_rotate_by_front_camera=pc_rotate_by_front_camera
        )
        self.task_offsets = _load_task_extrinsics_offsets()
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

    
    def _transform_pcd_to_front_frame(self, pcds, extrinsics):
        """
        Transform point clouds from world frame to front camera frame.
        
        Args:
            pcds: (B, ncam, 3, H, W) - point clouds in world coordinates
            extrinsics: (B, ncam, 4, 4) - camera-to-world transforms
            
        Returns:
            pcds_front: (B, ncam, 3, H, W) - point clouds in front camera frame
        """
        B, ncam, _, H, W = pcds.shape
        original_dtype = pcds.dtype
        
        # Get front camera extrinsics (camera 0 -> world)
        front_cam_to_world = extrinsics[:, 0:1].cuda(non_blocking=True).float()  # (B, 1, 4, 4)

        
        # Invert to get world -> front camera transform
        world_to_front = torch.linalg.inv(front_cam_to_world)  # (B, 1, 4, 4)
        
        # Reshape point clouds for matrix multiplication (convert to float for matmul)
        pcds_flat = pcds.float().reshape(B, ncam, 3, H * W)  # (B, ncam, 3, HW)
        
        # Add homogeneous coordinate
        ones = torch.ones(B, ncam, 1, H * W, device=pcds.device, dtype=torch.float32)
        pcds_homo = torch.cat([pcds_flat, ones], dim=2)  # (B, ncam, 4, HW)
        
        # Apply transformation: pcd_front = world_to_front @ pcd_world
        # Broadcast world_to_front across all cameras
        world_to_front = world_to_front.expand(B, ncam, 4, 4)  # (B, ncam, 4, 4)
        pcds_front_homo = torch.matmul(
            world_to_front.reshape(B * ncam, 4, 4),
            pcds_homo.reshape(B * ncam, 4, H * W)
        )  # (B*ncam, 4, HW)


        
        # Remove homogeneous coordinate and reshape back
        pcds_front = pcds_front_homo[:, :3].reshape(B, ncam, 3, H, W)
        
        # Convert back to original dtype

        return pcds_front.to(original_dtype)

    def _rotate_pcd_by_front_camera(self, pcds, extrinsics):
        """
        Rotate the point cloud by the front camera's extrinsic rotation only (no translation).
        Uses R^T from the front camera's camera-to-world transform so points are in the
        front camera's orientation.

        Args:
            pcds: (B, ncam, 3, H, W) - point clouds in world coordinates
            extrinsics: (B, ncam, 4, 4) - camera-to-world transforms

        Returns:
            pcds_rotated: (B, ncam, 3, H, W) - point clouds rotated by front camera R^T
        """
        B, ncam, _, H, W = pcds.shape
        original_dtype = pcds.dtype
        # Front camera extrinsic: camera-to-world, rotation is extrinsics[:, 0, :3, :3]
        R_c2w = extrinsics[:, 0:1, :3, :3].cuda(non_blocking=True).float()  # (B, 1, 3, 3)
        R_w2c = R_c2w.transpose(-1, -2)  # (B, 1, 3, 3), rotation only
        pcds_flat = pcds.float().reshape(B, ncam, 3, H * W)
        R_w2c = R_w2c.expand(B, ncam, 3, 3)
        rotated = torch.matmul(
            R_w2c.reshape(B * ncam, 3, 3),
            pcds_flat.reshape(B * ncam, 3, H * W)
        )
        return rotated.reshape(B, ncam, 3, H, W).to(original_dtype)

    def process_obs(self, rgbs, rgb2d, depth, extrinsics, intrinsics,
                    augment=False, task=None):
        """
        RGBs of shape (B, ncam, 3, h_i, w_i),
        depths of shape (B, ncam, h_i, w_i).
        Assume the 3d cameras go before 2d cameras.
        """
        # Get point cloud from depth (in world coordinates)
        pcds = self.depth2cloud(
            depth.cuda(non_blocking=True).to(torch.bfloat16),
            extrinsics.cuda(non_blocking=True).to(torch.bfloat16),
            intrinsics.cuda(non_blocking=True).to(torch.bfloat16)
        )

        # Handle non-wrist cameras, which may require augmentations
        if augment:
            b, nc, _, h, w = rgbs.shape
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
        
        # Optionally transform point clouds from world frame to front camera frame at the END
        # This is done after all augmentation and processing
        # Front camera is assumed to be index 0

        # Task-based extrinsics offset then transform to front camera frame
        if self.use_front_camera_frame:
            offsets = self.task_offsets
            extrinsics_for_frame = extrinsics.clone()
            for i in range(extrinsics.size(0)):
                task_name = task[i] if task is not None else None
                if task_name and task_name in offsets:
                    R, t = offsets[task_name]
                    extrinsics_for_frame[i, 0] = _apply_offset_to_extrinsics(
                        extrinsics_for_frame[i, 0], R, t,
                        extrinsics.device, extrinsics.dtype
                    )
            pcds = self._transform_pcd_to_front_frame(pcds, extrinsics_for_frame)

        if self.pc_rotate_by_front_camera:
            pcds = self._rotate_pcd_by_front_camera(pcds, extrinsics)

        
        return rgbs, pcds
