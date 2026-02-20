from kornia import augmentation as K
import time
import numpy as np
import torch
from torch.nn import functional as F

from .base import DataPreprocessor


class RLBenchDataPreprocessor(DataPreprocessor):

    def __init__(self, keypose_only=False, num_history=1,
                 orig_imsize=256, custom_imsize=None, depth2cloud=None,
                 use_front_camera_frame=False, pc_rotate_by_front_camera=False):
        super().__init__(
            keypose_only=keypose_only,
            num_history=num_history,
            custom_imsize=custom_imsize,
            depth2cloud=depth2cloud,
            use_front_camera_frame=use_front_camera_frame,
            pc_rotate_by_front_camera=pc_rotate_by_front_camera
        )
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

        # i have to do this for each elem in the batch separately. 
        # HACK
       
        if self.use_front_camera_frame:
            for i in range(extrinsics.size(0)):
                if task[i] == "bimanual_push_box":
                    pass
                elif task[i] == "bimanual_lift_tray":
                    # manually flip the extrinsics
                    extrinsics[i,0,0:3,3]= -extrinsics[i,0,0:3,3] # flip the translation. 
                    # rotate by 30 degress 
                    rotation_matrix = torch.tensor([[1, 0, 0], [0, np.cos(30*np.pi/180), -np.sin(30*np.pi/180)], [0, np.sin(30*np.pi/180), np.cos(30*np.pi/180)]]).to(extrinsics.device).to(extrinsics.dtype)
                    extrinsics[i,0,0:3,0:3] = torch.matmul(extrinsics[i,0,0:3,0:3], rotation_matrix)
            

            pcds = self._transform_pcd_to_front_frame(pcds, extrinsics)

        if self.pc_rotate_by_front_camera:
            pcds = self._rotate_pcd_by_front_camera(pcds, extrinsics)

        
        return rgbs, pcds
