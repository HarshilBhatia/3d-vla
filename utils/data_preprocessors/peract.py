from kornia import augmentation as K
import torch

from .base import DataPreprocessor


class PeractDataPreprocessor(DataPreprocessor):

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

    def process_obs(self, rgbs, pcds, augment=False):
        """
        RGBs of shape (B, ncam, 3, h_i, w_i),
        depths of shape (B, ncam, h_i, w_i).
        Assume the 3d cameras go before 2d cameras.
        """
        # Handle non-wrist cameras, which may require augmentations
        if augment:
            b, nc, _, h, w = rgbs.shape
            # Augment in half precision — fuse H2D + dtype cast into one copy_
            obs = torch.cat((
                rgbs.to(device='cuda', dtype=torch.float16, non_blocking=True) / 255,
                pcds.to(device='cuda', dtype=torch.float16, non_blocking=True)
            ), 2)  # (B, ncam, 6, H, W)
            obs = obs.reshape(-1, 6, h, w)
            obs = self.aug(obs)
            # Convert to full precision
            rgb_3d = obs[:, :3].reshape(b, nc, 3, h, w).float()
            pcd_3d = obs[:, 3:].reshape(b, nc, 3, h, w).float()
        else:
            # Simply convert to full precision — fuse H2D + dtype cast into one copy_
            rgb_3d = rgbs.to(device='cuda', dtype=torch.float32, non_blocking=True) / 255
            pcd_3d = pcds.to(device='cuda', dtype=torch.float32, non_blocking=True)

        return rgb_3d, pcd_3d
