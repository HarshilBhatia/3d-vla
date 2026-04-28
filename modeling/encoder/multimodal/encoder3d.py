import einops
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import Conv2dNormActivation

from ...utils.position_encodings import RotaryPositionEncoding3D, SinusoidalPosEmb
from ...utils.layers import AttentionModule
from ..vision.fpn import EfficientFeaturePyramidNetwork
from .base_encoder import Encoder as BaseEncoder


class Encoder(BaseEncoder):

    def __init__(self,
                 backbone="clip",
                 text_backbone=None,
                 embedding_dim=60,
                 nhist=1,
                 num_attn_heads=9,
                 num_vis_instr_attn_layers=2,
                 fps_subsampling_factor=5,
                 finetune_backbone=False,
                 finetune_text_encoder=False,
                 learn_extrinsics=False,
                 rope_type='normal'):
        super().__init__(
            backbone=backbone,
            text_backbone=text_backbone,
            embedding_dim=embedding_dim,
            nhist=nhist,
            num_attn_heads=num_attn_heads,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            finetune_text_encoder=finetune_text_encoder
        )
        
        # Store whether we're learning extrinsics (needed for gradient flow through RoPE)
        self.learn_extrinsics = learn_extrinsics

        # Postprocess scene features
        if self._backbone_name == 'clip':
            # self.backbone.to(memory_format=torch.channels_last)
            self.output_level = "res3"
            self.feature_pyramid = EfficientFeaturePyramidNetwork(
                [64, 256, 512, 1024, 2048],
                embedding_dim, output_level="res3"
            )
            self.rgb2d_proj = nn.Linear(1024, embedding_dim)
        elif self._backbone_name == 'siglip2':
            self.siglip2_proj = nn.Conv2d(self.backbone.hidden_size, embedding_dim, kernel_size=1)
        elif self._backbone_name == 'dino':
            self.dino_proj = nn.Conv2d(self.backbone.hidden_size, embedding_dim, kernel_size=1)

        # 3D relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim, rope_type=rope_type)

        # Proprioception learnable encoding if 3D is used
        self.curr_gripper_embed = nn.Embedding(nhist, embedding_dim)
        self.gripper_context_head = AttentionModule(
            num_layers=3, d_model=embedding_dim, dim_fw=embedding_dim,
            n_heads=num_attn_heads, rotary_pe=True, use_adaln=False,
            pre_norm=False
        )

        # Camera IDs for the 2D cameras
        self.camera_ids = nn.Embedding(2, embedding_dim)
        self.pos_embed_2d = SinusoidalPosEmb(embedding_dim)

    def encode_proprio(self, proprio, context_feats, context_pos, stopgrad_k=0):
        """
        Compute proprioception features.

        Args:
            - proprio: (B, nhist, 3+)
            - context_feats: (B, npt, C)
            - context_pos: (B, npt, 3)
            - stopgrad_k: number of bins to zero out in backward (for RoPE stopgrad)

        Returns:
            - gripper_feats: (B, nhist, F)
        """
        # Learnable embedding for proprioception
        proprio_feats = self.curr_gripper_embed.weight.unsqueeze(0).repeat(
            len(proprio), 1, 1
        )

        # Rotary positional encoding
        proprio_pos = self.relative_pe_layer(proprio[..., :3], stopgrad_k=stopgrad_k)
        # Allow gradients for context_pos if learning extrinsics
        # This is needed because point cloud positions depend on learned camera parameters
        # allow_grad = self.training and self.learn_extrinsics
        context_pos = self.relative_pe_layer(context_pos, allow_grad=False, stopgrad_k=stopgrad_k) # this is to encode the proprio, don't need to backprop here.

        # Attention to scene tokens
        proprio_feats = self.gripper_context_head(
            proprio_feats, context_feats,
            seq1_pos=proprio_pos, seq2_pos=context_pos
        )[-1]

        return proprio_feats

    def encode_clip(self, rgb3d, rgb2d, pcd, text):
        """
        Compute visual features/pos embeddings.

        Args:
            - rgb3d: (B, ncam3d, 3, H, W) or (B, nhist, ncam3d, 3, H, W)
            - pcd: same spatial shape as rgb3d, xyz channels

        Returns:
            - rgb3d_feats: (B, Np, F) or (B, nhist, Np, F)
            - pcd: (B, Np, 3) or (B, nhist, Np, 3)
            - instr_feats: (B, L, F)
        """
        # Encode language
        instruction = self.text_encoder(text)
        instr_feats = self.instruction_encoder(instruction)  # (B, L, F)

        has_hist = rgb3d.ndim == 6
        if has_hist:
            B, nhist, ncam_rgb, C, H, W = rgb3d.shape
            ncam_pcd = pcd.shape[2]
            rgb3d = rgb3d.view(B * nhist, ncam_rgb, C, H, W)
            pcd   = pcd.view(B * nhist, ncam_pcd, 3, *pcd.shape[-2:])
            instr_exp = instr_feats.unsqueeze(1).expand(-1, nhist, -1, -1).reshape(
                B * nhist, instr_feats.shape[1], instr_feats.shape[2]
            )
        else:
            instr_exp = instr_feats

        # 3D camera features
        num_cameras = rgb3d.shape[1]
        rgb3d = einops.rearrange(rgb3d, "bt ncam c h w -> (bt ncam) c h w")
        rgb3d = self.normalize(rgb3d).contiguous()
        rgb3d_feats = self.backbone(rgb3d)
        # print(rgb3d.dtype, self.backbone.dtype)
        rgb3d_feats = self.feature_pyramid(rgb3d_feats)[self.output_level]
        feat_h, feat_w = rgb3d_feats.shape[-2:]
        rgb3d_feats = einops.rearrange(
            rgb3d_feats, "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
        )
        # Attention from vision to language
        rgb3d_feats = self.vl_attention(seq1=rgb3d_feats, seq2=instr_exp)[-1]

        # Point cloud
        num_cameras_pcd = pcd.shape[1]
        pcd = F.interpolate(
            einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w"),
            (feat_h, feat_w),
            mode='bilinear'
        )
        pcd = einops.rearrange(
            pcd, "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras_pcd
        )

        if has_hist:
            rgb3d_feats = rgb3d_feats.view(B, nhist, *rgb3d_feats.shape[1:])
            pcd = pcd.view(B, nhist, *pcd.shape[1:])

        rgb2d_feats = None
        return rgb3d_feats, rgb2d_feats, pcd, instr_feats

    def encode_siglip2(self, rgb3d, rgb2d, pcd, text):
        """
        Compute visual features/pos embeddings using SigLIP2.
        Supports rgb3d of shape (B, ncam3d, 3, H, W) or (B, nhist, ncam3d, 3, H, W).
        """
        # Encode language
        instruction = self.text_encoder(text)
        instr_feats = self.instruction_encoder(instruction)  # (B, L, F)

        has_hist = rgb3d.ndim == 6
        if has_hist:
            B, nhist, ncam_rgb, C, H, W = rgb3d.shape
            ncam_pcd = pcd.shape[2]
            rgb3d = rgb3d.view(B * nhist, ncam_rgb, C, H, W)
            pcd   = pcd.view(B * nhist, ncam_pcd, 3, *pcd.shape[-2:])
            instr_exp = instr_feats.unsqueeze(1).expand(-1, nhist, -1, -1).reshape(
                B * nhist, instr_feats.shape[1], instr_feats.shape[2]
            )
        else:
            instr_exp = instr_feats

        # 3D camera features
        num_cameras = rgb3d.shape[1]
        rgb3d = einops.rearrange(rgb3d, "bt ncam c h w -> (bt ncam) c h w")
        rgb3d = self.normalize(rgb3d)
        rgb3d_feats = self.backbone(rgb3d)            # (bt*ncam, hidden_size, h, w)
        rgb3d_feats = self.siglip2_proj(rgb3d_feats)  # (bt*ncam, F, h, w)
        feat_h, feat_w = rgb3d_feats.shape[-2:]
        rgb3d_feats = einops.rearrange(
            rgb3d_feats, "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
        )
        # Attention from vision to language
        rgb3d_feats = self.vl_attention(seq1=rgb3d_feats, seq2=instr_exp)[-1]

        # Point cloud: interpolate to ViT spatial resolution
        num_cameras_pcd = pcd.shape[1]
        pcd = F.interpolate(
            einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w"),
            (feat_h, feat_w),
            mode='bilinear'
        )
        pcd = einops.rearrange(
            pcd, "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras_pcd
        )

        if has_hist:
            rgb3d_feats = rgb3d_feats.view(B, nhist, *rgb3d_feats.shape[1:])
            pcd = pcd.view(B, nhist, *pcd.shape[1:])

        rgb2d_feats = None
        return rgb3d_feats, rgb2d_feats, pcd, instr_feats

    def encode_dino(self, rgb3d, rgb2d, pcd, text):
        """
        Compute visual features/pos embeddings using DINOv2 + CLIP text.
        Supports rgb3d of shape (B, ncam3d, 3, H, W) or (B, nhist, ncam3d, 3, H, W).
        """
        # Encode language
        instruction = self.text_encoder(text)
        instr_feats = self.instruction_encoder(instruction)  # (B, L, F)

        has_hist = rgb3d.ndim == 6
        if has_hist:
            B, nhist, ncam_rgb, C, H, W = rgb3d.shape
            ncam_pcd = pcd.shape[2]
            rgb3d = rgb3d.view(B * nhist, ncam_rgb, C, H, W)
            pcd   = pcd.view(B * nhist, ncam_pcd, 3, *pcd.shape[-2:])
            instr_exp = instr_feats.unsqueeze(1).expand(-1, nhist, -1, -1).reshape(
                B * nhist, instr_feats.shape[1], instr_feats.shape[2]
            )
        else:
            instr_exp = instr_feats

        # 3D camera features
        num_cameras = rgb3d.shape[1]
        rgb3d = einops.rearrange(rgb3d, "bt ncam c h w -> (bt ncam) c h w")
        rgb3d = self.normalize(rgb3d)
        rgb3d_feats = self.backbone(rgb3d)           # (bt*ncam, hidden_size, h, w)
        rgb3d_feats = self.dino_proj(rgb3d_feats)    # (bt*ncam, F, h, w)
        feat_h, feat_w = rgb3d_feats.shape[-2:]
        rgb3d_feats = einops.rearrange(
            rgb3d_feats, "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
        )
        # Attention from vision to language
        rgb3d_feats = self.vl_attention(seq1=rgb3d_feats, seq2=instr_exp)[-1]

        # Point cloud: interpolate to DINOv2 spatial resolution
        num_cameras_pcd = pcd.shape[1]
        pcd = F.interpolate(
            einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w"),
            (feat_h, feat_w),
            mode='bilinear'
        )
        pcd = einops.rearrange(
            pcd, "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras_pcd
        )

        if has_hist:
            rgb3d_feats = rgb3d_feats.view(B, nhist, *rgb3d_feats.shape[1:])
            pcd = pcd.view(B, nhist, *pcd.shape[1:])

        return rgb3d_feats, None, pcd, instr_feats
