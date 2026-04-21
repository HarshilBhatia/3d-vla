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
                 text_backbone=None,
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
                 sa_blocks_use_rope=True,
                 predict_extrinsics=True,
                 extrinsics_prediction_mode='delta_m',
                 # RoPE type
                 rope_type='adam',
                 dynamic_rope_from_camtoken=False):
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
            traj_scene_rope=traj_scene_rope,
            sa_blocks_use_rope=sa_blocks_use_rope,
            learn_extrinsics=learn_extrinsics,
            predict_extrinsics=predict_extrinsics,
            extrinsics_prediction_mode=extrinsics_prediction_mode,
        )


        print(f'learn_extrinsics: {learn_extrinsics}')
        print(f'predict_extrinsics: {predict_extrinsics}')
        print(f'extrinsics_prediction_mode: {extrinsics_prediction_mode}')
        print(f'rope_type: {rope_type}')
        
        # Vision-language encoder, runs only once
        self.encoder = Encoder(
            backbone=backbone,
            text_backbone=text_backbone,
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            finetune_text_encoder=finetune_text_encoder,
            learn_extrinsics=learn_extrinsics,
            rope_type=rope_type
        )

        # Action decoder, runs at every denoising timestep
        self.prediction_head = TransformerHead(
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            learn_extrinsics=learn_extrinsics,
            traj_scene_rope=traj_scene_rope,
            sa_blocks_use_rope=sa_blocks_use_rope,
            predict_extrinsics=predict_extrinsics,
            extrinsics_prediction_mode=extrinsics_prediction_mode,
            rope_type=rope_type,
            dynamic_rope_from_camtoken=dynamic_rope_from_camtoken,
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
    
    def encode_inputs(self, rgb3d, rgb2d, pcd, instruction, proprio, stopgrad_k=0):
        """Override to apply learned transformation to point cloud."""
        # Apply learned camera-to-world transformation if enabled
        pcd = self.transform_pcd_to_world(pcd)
        
        # Call parent's encode_inputs with transformed point cloud
        return super().encode_inputs(rgb3d, rgb2d, pcd, instruction, proprio, stopgrad_k=stopgrad_k)



def _transform_pcd_with_extrinsics(pcd, cam_params):

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

class TransformerHead(BaseTransformerHead):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 nhist=3,
                 num_shared_attn_layers=4,
                 rotary_pe=True,
                 learn_extrinsics=False,
                 traj_scene_rope=True,
                 sa_blocks_use_rope=True,
                 predict_extrinsics=True,
                 rope_type='normal',
                 **kwargs):
        super().__init__(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            nhist=nhist,
            num_shared_attn_layers=num_shared_attn_layers,
            rotary_pe=rotary_pe,
            traj_scene_rope=traj_scene_rope,
            sa_blocks_use_rope=sa_blocks_use_rope,
            predict_extrinsics=predict_extrinsics,
            learn_extrinsics=learn_extrinsics,
            extrinsics_prediction_mode=kwargs.get("extrinsics_prediction_mode", 'delta_m'),
            dynamic_rope_from_camtoken=kwargs.get("dynamic_rope_from_camtoken", False),
        )

        # Store whether we're learning extrinsics (needed for gradient flow through RoPE)
        self.learn_extrinsics = learn_extrinsics
        self.predict_extrinsics = predict_extrinsics

        # Relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim, rope_type=rope_type)


    def get_positional_embeddings(
        self,
        traj_xyz, traj_feats,
        rgb3d_pos, rgb3d_feats, rgb2d_feats, rgb2d_pos,
        timesteps, proprio_feats,
        fps_scene_feats, fps_scene_pos,
        instr_feats, instr_pos,
        stopgrad_k=0,
        delta_M=None,
        cam_params_rt=None,
        fps_cam_ids=None,
    ):
        # RT mode: transform positions by predicted R,t (camera -> world); no delta_M in RoPE.
        # delta_M mode: no position transform; delta_M mixes sin/cos in RoPE.
        allow_grad = self.training and (self.learn_extrinsics or self.predict_extrinsics)

        if cam_params_rt is not None:
            # Predict R,T: transform scene positions to world frame; RoPE sees transformed positions, no delta_M
            rgb3d_pos = _transform_pcd_with_extrinsics(rgb3d_pos, cam_params_rt)
            fps_scene_pos = _transform_pcd_with_extrinsics(fps_scene_pos, cam_params_rt)
            delta_M = None

        # Expand per-camera delta_M (B, ncam, 6, 6) to per-token for rgb3d and fps sequences
        delta_M_rgb3d = delta_M
        delta_M_fps = delta_M
        if delta_M is not None and delta_M.ndim == 4 and fps_cam_ids is not None:
            B, ncam = delta_M.shape[:2]
            Np = rgb3d_pos.shape[1]
            P = Np // ncam  # tokens per camera in the dense sequence

            # Dense rgb3d tokens: cam index = token_index // P
            dense_cam_ids = torch.arange(ncam, device=delta_M.device).repeat_interleave(P)  # (Np,)
            delta_M_rgb3d = delta_M[:, dense_cam_ids, :, :]  # (B, Np, 6, 6)

            # FPS tokens: first M from fps_cam_ids, last ncam are per-image tokens (cam 0..ncam-1)
            M = fps_cam_ids.shape[1]
            delta_M_fps_sparse = delta_M[torch.arange(B, device=delta_M.device)[:, None], fps_cam_ids]  # (B, M, 6, 6) or (B, M, D, D)
            delta_M_fps = torch.cat([delta_M_fps_sparse, delta_M], dim=1)  # (B, M+ncam, 6, 6) or (B, M+ncam, D, D)

        rel_traj_pos = self.relative_pe_layer(traj_xyz, stopgrad_k=stopgrad_k)
        rel_scene_pos = self.relative_pe_layer(
            rgb3d_pos,
            allow_grad=allow_grad,
            stopgrad_k=stopgrad_k,
            delta_M=delta_M_rgb3d,
        )

        rel_fps_pos = self.relative_pe_layer(
            fps_scene_pos,
            allow_grad=allow_grad,
            stopgrad_k=stopgrad_k,
            delta_M=delta_M_fps,
        )
        
        # Add zero positional embeddings for register tokens (4) and camera token (1)
        batch_size = traj_xyz.shape[0]
        num_additional_tokens = 5  # 4 register + 1 camera
        zero_pos = torch.zeros(batch_size, num_additional_tokens, rel_traj_pos.shape[-2], rel_traj_pos.shape[-1],  
                              device=rel_traj_pos.device, dtype=rel_traj_pos.dtype)
        
        rel_pos = torch.cat([rel_traj_pos, rel_fps_pos, zero_pos], 1)
        return rel_traj_pos, rel_scene_pos, rel_pos, rel_fps_pos

    def _precompute_rope_bases(self, traj_xyz, rgb3d_pos, fps_scene_pos, stopgrad_k):
        """Pre-compute sin/cos bases for traj, scene, and fps positions (delta_M mode only).

        Returns (traj_base, scene_base, fps_base), each [B, N, d//6, 6], detached.
        Called once before the per-block loop; bases are reused with different delta_M each block.
        """
        traj_base = self.relative_pe_layer._compute_sincos_base(traj_xyz, stopgrad_k)
        scene_base = self.relative_pe_layer._compute_sincos_base(rgb3d_pos, stopgrad_k)
        fps_base = self.relative_pe_layer._compute_sincos_base(fps_scene_pos, stopgrad_k)
        return traj_base, scene_base, fps_base

    def _apply_delta_M_rope(self, traj_base, scene_base, fps_base, delta_M, delta_M_fps=None):
        """Apply delta_M to pre-computed sin/cos bases and return RoPE positions.

        Traj uses no delta_M. Scene gets delta_M_scene, fps gets delta_M_fps (defaults to delta_M).

        Returns (rel_traj_pos, rel_scene_pos, rel_pos, rel_fps_pos).
        """
        if delta_M_fps is None:
            delta_M_fps = delta_M
        rel_traj_pos = self.relative_pe_layer._finalize_from_base(traj_base, delta_M=None)
        rel_scene_pos = self.relative_pe_layer._finalize_from_base(scene_base, delta_M=delta_M)
        rel_fps_pos = self.relative_pe_layer._finalize_from_base(fps_base, delta_M=delta_M_fps)

        batch_size = traj_base.shape[0]
        zero_pos = torch.zeros(
            batch_size, 5, rel_traj_pos.shape[-2], rel_traj_pos.shape[-1],
            device=traj_base.device, dtype=rel_traj_pos.dtype)
        rel_pos = torch.cat([rel_traj_pos, rel_fps_pos, zero_pos], 1)

        return rel_traj_pos, rel_scene_pos, rel_pos, rel_fps_pos

    def _recompute_rope(self, cam_feat, traj_xyz, orig_rgb3d_pos, orig_fps_scene_pos, stopgrad_k,
                        bases=None, fps_cam_ids=None, per_img_feats=None):
        """
        Predict delta_M or (R,T) and recompute 3D RoPE embeddings.

        Args:
            cam_feat: (B, C) — unused in delta_M mode; kept for RT mode
            traj_xyz: (B, T, 3)
            orig_rgb3d_pos: (B, N, 3)
            orig_fps_scene_pos: (B, M+ncam, 3)
            stopgrad_k: int
            bases: optional (traj_base, scene_base, fps_base) from _precompute_rope_bases
            fps_cam_ids: (B, M) — required; camera index per fps token
            per_img_feats: (B, ncam, C) — required; current per-image avg token features

        Returns:
            (rel_traj_pos, rel_scene_pos, rel_pos, rel_fps_pos)
        """
        allow_grad = self.training and (self.learn_extrinsics or self.predict_extrinsics)

        assert fps_cam_ids is not None and per_img_feats is not None, \
            "_recompute_rope requires fps_cam_ids and per_img_feats"

        cam_params_rt, delta_M = self._predict_from_cam_feat(per_img_feats)  # (B, ncam, 6/D, 6/D)

        if cam_params_rt is not None:
            rgb3d_pos = _transform_pcd_with_extrinsics(orig_rgb3d_pos, cam_params_rt)
            fps_scene_pos = _transform_pcd_with_extrinsics(orig_fps_scene_pos, cam_params_rt)
            delta_M = None
        else:
            rgb3d_pos = orig_rgb3d_pos
            fps_scene_pos = orig_fps_scene_pos

        # Expand per-camera delta_M (B, ncam, ...) to per-token for rgb3d and fps
        delta_M_rgb3d = delta_M
        delta_M_fps = delta_M
        if delta_M is not None and delta_M.ndim >= 4:
            B, ncam = delta_M.shape[:2]
            Np = orig_rgb3d_pos.shape[1]
            P = Np // ncam
            dense_cam_ids = torch.arange(ncam, device=delta_M.device).repeat_interleave(P)
            delta_M_rgb3d = delta_M[:, dense_cam_ids, :, :]  # (B, Np, 6, 6) or (B, Np, D, D)
            M = fps_cam_ids.shape[1]
            delta_M_fps_sparse = delta_M[torch.arange(B, device=delta_M.device)[:, None], fps_cam_ids]
            delta_M_fps = torch.cat([delta_M_fps_sparse, delta_M], dim=1)  # (B, M+ncam, ...)

        # delta_M mode with pre-computed bases: skip sin/cos recomputation
        if delta_M is not None and bases is not None:
            traj_base, scene_base, fps_base = bases
            rel_traj_pos, rel_scene_pos, rel_pos, rel_fps_pos = self._apply_delta_M_rope(
                traj_base, scene_base, fps_base, delta_M_rgb3d, delta_M_fps)
        else:
            rel_traj_pos = self.relative_pe_layer(traj_xyz, stopgrad_k=stopgrad_k)
            rel_scene_pos = self.relative_pe_layer(
                rgb3d_pos, allow_grad=allow_grad, stopgrad_k=stopgrad_k, delta_M=delta_M_rgb3d)
            rel_fps_pos = self.relative_pe_layer(
                fps_scene_pos, allow_grad=allow_grad, stopgrad_k=stopgrad_k, delta_M=delta_M_fps)

            batch_size = traj_xyz.shape[0]
            zero_pos = torch.zeros(
                batch_size, 5, rel_traj_pos.shape[-2], rel_traj_pos.shape[-1],
                device=traj_xyz.device, dtype=rel_traj_pos.dtype)
            rel_pos = torch.cat([rel_traj_pos, rel_fps_pos, zero_pos], 1)

        # Update last predicted cam params for W&B logging (last prediction wins)
        last_pred = cam_params_rt if cam_params_rt is not None else delta_M
        self._last_predicted_cam_params = last_pred.detach()

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
