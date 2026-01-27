import torch
from torch import nn

from ..encoder.multimodal.encoder3d import Encoder
from ..utils.position_encodings import RotaryPositionEncoding3D
from utils.pytorch3d_transforms import axis_angle_to_matrix

from .base_denoise_actor import DenoiseActor as BaseDenoiseActor
from .base_denoise_actor import TransformerHead as BaseTransformerHead


class DenoiseActor(BaseDenoiseActor):

    def __init__(self,
                 # Encoder arguments
                 backbone="clip",
                 finetune_backbone=False,
                 finetune_text_encoder=False,
                 num_vis_instr_attn_layers=2,
                 fps_subsampling_factor=5,
                 # Encoder and decoder arguments
                 embedding_dim=60,
                 num_attn_heads=9,
                 nhist=3,
                 nhand=1,
                 # Decoder arguments
                 num_shared_attn_layers=4,
                 relative=False,
                 rotation_format='quat_xyzw',
                 # Denoising arguments
                 denoise_timesteps=100,
                 denoise_model="ddpm",
                 # Training arguments
                 lv2_batch_size=1,
                 # Learnable extrinsics (camera -> world)
                 learn_extrinsics=False):
        super().__init__(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            nhist=nhist,
            nhand=nhand,
            num_shared_attn_layers=num_shared_attn_layers,
            relative=relative,
            rotation_format=rotation_format,
            denoise_timesteps=denoise_timesteps,
            denoise_model=denoise_model,
            lv2_batch_size=lv2_batch_size
        )

        # Vision-language encoder, runs only once
        self.encoder = Encoder(
            backbone=backbone,
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            finetune_text_encoder=finetune_text_encoder
        )

        # Action decoder, runs at every denoising timestep
        self.prediction_head = TransformerHead(
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers
        )
        
        # Learnable camera extrinsics: axis-angle (3) + translation (3) = 6 params
        self.learn_extrinsics = learn_extrinsics
        if learn_extrinsics:
            # Initialize to identity transform
            self.cam_axis_angle = nn.Parameter(torch.randn(3))  # rotation
            self.cam_translation = nn.Parameter(torch.randn(3))  # translation
    
    
    def get_learned_extrinsics(self):
        """Convert 6D params to 4x4 extrinsics matrix."""
        if not self.learn_extrinsics:
            return None
        
        # Convert axis-angle to rotation matrix
        R = axis_angle_to_matrix(self.cam_axis_angle.unsqueeze(0))[0]  # (3, 3)
        t = self.cam_translation.unsqueeze(1)  # (3, 1)
        
        # Build 4x4 extrinsics matrix
        extrinsics = torch.eye(4, device=R.device, dtype=R.dtype)
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = t.squeeze()
        
        return extrinsics  # (4, 4)
    
    def transform_pcd_to_world(self, pcd_cam):
        """
        Transform point cloud from camera frame to world frame.
        
        Args:
            pcd_cam: (B, ncam, 3, H, W) - points in camera coordinates
            
        Returns:
            pcd_world: (B, ncam, 3, H, W) - points in world coordinates
        """
        if not self.learn_extrinsics:
            return pcd_cam
        
        extrinsics = self.get_learned_extrinsics()  # (4, 4)
        
        B, ncam, _, H, W = pcd_cam.shape
        
        # Reshape for transformation
        pcd_flat = pcd_cam.reshape(B * ncam, 3, H * W)  # (B*ncam, 3, HW)
        
        # Extract R and t
        R = extrinsics[:3, :3]  # (3, 3)
        t = extrinsics[:3, 3:4]  # (3, 1)
        
        # Transform: P_world = R @ P_cam + t
        pcd_world_flat = R @ pcd_flat + t  # (B*ncam, 3, HW)
        
        # Reshape back
        pcd_world = pcd_world_flat.reshape(B, ncam, 3, H, W)
        
        return pcd_world
    
    def encode_inputs(self, rgb3d, rgb2d, pcd, instruction, proprio):
        """Override to apply learned transformation to point cloud."""
        # Apply learned camera-to-world transformation if enabled
        pcd = self.transform_pcd_to_world(pcd)
        
        # Store flags for encoder and decoder to allow gradients through RoPE
        self.encoder._allow_pe_grad = self.learn_extrinsics
        self.prediction_head._allow_pe_grad = self.learn_extrinsics
        
        # Call parent's encode_inputs with transformed point cloud
        outputs = super().encode_inputs(rgb3d, rgb2d, pcd, instruction, proprio)
        
        # Clean up flags
        self.encoder._allow_pe_grad = False
        self.prediction_head._allow_pe_grad = False
        
        return outputs


class TransformerHead(BaseTransformerHead):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 nhist=3,
                 num_shared_attn_layers=4,
                 rotary_pe=True):
        super().__init__(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            nhist=nhist,
            num_shared_attn_layers=num_shared_attn_layers,
            rotary_pe=rotary_pe
        )

        # Relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

    def get_positional_embeddings(
        self,
        traj_xyz, traj_feats,
        rgb3d_pos, rgb3d_feats, rgb2d_feats, rgb2d_pos,
        timesteps, proprio_feats,
        fps_scene_feats, fps_scene_pos,
        instr_feats, instr_pos
    ):
        # Check if parent model is learning extrinsics (set via encode_inputs)
        allow_grad = getattr(self, '_allow_pe_grad', False)
        
        rel_traj_pos = self.relative_pe_layer(traj_xyz)
        rel_scene_pos = self.relative_pe_layer(rgb3d_pos, allow_grad=allow_grad)
        rel_fps_pos = self.relative_pe_layer(fps_scene_pos, allow_grad=allow_grad)
        rel_pos = torch.cat([rel_traj_pos, rel_fps_pos], 1)
        return rel_traj_pos, rel_scene_pos, rel_pos

    def get_sa_feature_sequence(
        self,
        traj_feats, fps_scene_feats,
        rgb3d_feats, rgb2d_feats, instr_feats
    ):
        features = torch.cat([traj_feats, fps_scene_feats], 1)
        return features
