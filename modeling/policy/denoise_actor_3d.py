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
                 learn_extrinsics=False,
                 traj_scene_rope=True,
                 predict_extrinsics=True):
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
            lv2_batch_size=lv2_batch_size,
            traj_scene_rope=traj_scene_rope
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
            finetune_text_encoder=finetune_text_encoder,
            learn_extrinsics=learn_extrinsics
        )

        # Action decoder, runs at every denoising timestep
        self.prediction_head = TransformerHead(
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            learn_extrinsics=learn_extrinsics,
            traj_scene_rope=traj_scene_rope,
            predict_extrinsics=predict_extrinsics
        )
        
        # Learnable camera extrinsics: axis-angle (3) + translation (3) = 6 params
        self.learn_extrinsics = learn_extrinsics
        if learn_extrinsics:

            raise NotImplementedError("Should NOT be USED HERE")
            # Initialize to identity transform

            # randn if not rotated in data processing 
            # self.cam_axis_angle = nn.Parameter(torch.randn(3))  # rotation
            # self.cam_translation = nn.Parameter(torch.randn(3))  # translation

            # identity (better)
            self.cam_axis_angle = nn.Parameter(torch.zeros(3))  # rotation
            self.cam_translation = nn.Parameter(torch.zeros(3))  # translation
    
    
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
        
        # Call parent's encode_inputs with transformed point cloud
        return super().encode_inputs(rgb3d, rgb2d, pcd, instruction, proprio)


class TransformerHead(BaseTransformerHead):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 nhist=3,
                 num_shared_attn_layers=4,
                 rotary_pe=True,
                 learn_extrinsics=False,
                 traj_scene_rope=True,
                 predict_extrinsics=True):
        super().__init__(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            nhist=nhist,
            num_shared_attn_layers=num_shared_attn_layers,
            rotary_pe=rotary_pe,
            traj_scene_rope=traj_scene_rope,
            predict_extrinsics=predict_extrinsics
        )

        # Store whether we're learning extrinsics (needed for gradient flow through RoPE)
        self.learn_extrinsics = learn_extrinsics
        self.predict_extrinsics = predict_extrinsics
        
        # Relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

    def transform_pcd_with_extrinsics(self, pcd, cam_params):
        """
        Transform point cloud using predicted camera extrinsics.
        
        Args:
            pcd: (B, N, 3) - point cloud in camera coordinates
            cam_params: (B, 6) - camera extrinsics (3 axis-angle + 3 translation)
            
        Returns:
            pcd_transformed: (B, N, 3) - transformed point cloud in world coordinates
        """
        B, N, _ = pcd.shape
        
        # Extract axis-angle and translation
        axis_angle = cam_params[:, :3]  # (B, 3)
        translation = cam_params[:, 3:6]  # (B, 3)
        
        # Convert axis-angle to rotation matrix
        R = axis_angle_to_matrix(axis_angle)  # (B, 3, 3)
        
        # Transform: P_world = R @ P_cam + t
        # pcd: (B, N, 3) -> (B, 3, N) for matrix multiplication
        pcd_transposed = pcd.transpose(1, 2)  # (B, 3, N)
        pcd_rotated = torch.bmm(R, pcd_transposed)  # (B, 3, N)
        pcd_transformed = pcd_rotated.transpose(1, 2) + translation.unsqueeze(1)  # (B, N, 3)
        
        return pcd_transformed

    def get_positional_embeddings(
        self,
        traj_xyz, traj_feats,
        rgb3d_pos, rgb3d_feats, rgb2d_feats, rgb2d_pos,
        timesteps, proprio_feats,
        fps_scene_feats, fps_scene_pos,
        instr_feats, instr_pos
    ):
        # Allow gradients through RoPE when learning extrinsics
        # This is needed because the point cloud positions depend on learned camera parameters
        allow_grad = self.training
        
        rel_traj_pos = self.relative_pe_layer(traj_xyz)
        rel_scene_pos = self.relative_pe_layer(rgb3d_pos)

        # because absolute positions are used, it makes sense to concatenate. 
        rel_fps_pos = self.relative_pe_layer(fps_scene_pos, allow_grad=allow_grad) # only place where it makes sense to backprop
        
        # Add zero positional embeddings for register tokens (4) and camera token (1)
        batch_size = traj_xyz.shape[0]
        num_additional_tokens = 5  # 4 register + 1 camera
        zero_pos = torch.zeros(batch_size, num_additional_tokens, rel_traj_pos.shape[-2], rel_traj_pos.shape[-1],  
                              device=rel_traj_pos.device, dtype=rel_traj_pos.dtype)
        
        rel_pos = torch.cat([rel_traj_pos, rel_fps_pos, zero_pos], 1)
        return rel_traj_pos, rel_scene_pos, rel_pos, rel_fps_pos

    def get_sa_feature_sequence(
        self,
        traj_feats, fps_scene_feats,
        rgb3d_feats, rgb2d_feats, instr_feats
    ):
        batch_size = traj_feats.shape[0]
        
        # Expand learnable tokens to batch size
        register_tokens = self.register_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        camera_token = self.camera_token.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate: trajectory, scene, register tokens, camera token
        features = torch.cat([traj_feats, fps_scene_feats, register_tokens, camera_token], 1)
        return features
