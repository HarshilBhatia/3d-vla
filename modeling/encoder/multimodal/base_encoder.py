import einops
import torch
from torch import nn

from ...utils.layers import AttentionModule
from ..vision import fetch_visual_encoders
from ..text import fetch_text_encoders


class Encoder(nn.Module):

    def __init__(self,
                 backbone="clip",
                 text_backbone=None,
                 embedding_dim=60,
                 nhist=1,
                 num_attn_heads=9,
                 num_vis_instr_attn_layers=2,
                 fps_subsampling_factor=5,
                 finetune_backbone=False,
                 finetune_text_encoder=False):
        super().__init__()
        self.subsampling_factor = fps_subsampling_factor
        self._backbone_name = backbone
        # text_backbone defaults to backbone for backward compatibility
        self._text_backbone_name = text_backbone if text_backbone is not None else backbone

        # Instruction encoder
        self.text_encoder, _dim = fetch_text_encoders(self._text_backbone_name)
        if self.text_encoder is not None:  # is None when using a VLM
            for p in self.text_encoder.parameters():
                p.requires_grad = finetune_text_encoder
            self.instruction_encoder = nn.Linear(_dim, embedding_dim)

        # Scene encoder
        self.backbone, self.normalize = fetch_visual_encoders(backbone)
        for p in self.backbone.parameters():
            p.requires_grad = finetune_backbone

        # Attention from vision to language
        if backbone in ('clip', 'siglip2', 'dino'):
            self.vl_attention = AttentionModule(
                num_layers=num_vis_instr_attn_layers, d_model=embedding_dim,
                dim_fw=4 * embedding_dim, n_heads=num_attn_heads, pre_norm=False
            )

    def forward(self, rgb3d, rgb2d, pcd, instruction, proprio, stopgrad_k=0):
        """
        Encode different modalities, independent of denoising step.

        Args:
            - rgb3d: (B, ncam3d, 3, H, W)
            - rgb2d: (B, ncam2d, 3, H, W)
            - pcd: (B, ncam3d, 3, H, W)
            - instruction: (B, nt), tokens
            - proprio: (B, nhist, 3+6+X)
            - stopgrad_k: number of bins to zero out in backward (for RoPE stopgrad)

        Returns:
            - rgb3d_feats: (B, N, F)
            - pcd: (B, N, 3)
            - rgb2d_feats: (B, N2d, F)
            - rgb2d_pos: (B, N2d, 3)
            - instr_feats: (B, L, F)
            - instr_pos: (B, L, 3)
            - proprio_feats: (B, nhist, F)
            - fps_scene_feats: (B, M+ncam, F) — M fps tokens then ncam per-image avg tokens
            - fps_scene_pos: (B, M+ncam, 3)
            - fps_cam_ids: (B, M) — camera index for each fps token; per-image token for cam c is at M+c
        """
        vl_enc_fn = {
            'clip': self.encode_clip,
            'siglip2': self.encode_siglip2,
            'dino': self.encode_dino,
        }[self._backbone_name]
        # Compute scene features/positional embeddings, language embeddings
        rgb3d_feats, rgb2d_feats, pcd, instr_feats = vl_enc_fn(
            rgb3d, rgb2d, pcd, instruction
        )
        rgb2d_pos = None

        # Use the current end-effector position as language 'position'
        instr_pos = proprio[:, -1:, :3].repeat(1, instr_feats.size(1), 1)

        # Encode proprioception
        proprio_feats = self.encode_proprio(proprio, rgb3d_feats, pcd, stopgrad_k=stopgrad_k)

        # Build (B, Np) camera-index tensor: tokens are ordered [cam0 × P, cam1 × P, ...]
        ncam = rgb3d.shape[1]
        Np = rgb3d_feats.shape[1]
        cam_ids_full = (
            torch.arange(ncam, device=rgb3d_feats.device)
            .repeat_interleave(Np // ncam)                    # (Np,)
            .unsqueeze(0).expand(rgb3d_feats.shape[0], -1)   # (B, Np)
        )

        # Point subsampling based on scene features; also subsample cam_ids
        fps_scene_feats, fps_scene_pos, fps_cam_ids = self.run_dps(rgb3d_feats, pcd, cam_ids_full)

        # Per-image average tokens: one token per camera (B, ncam, F) / (B, ncam, 3)
        # Camera c's average token is at index fps_scene_feats.shape[1] + c after concat
        per_img_feats = einops.rearrange(rgb3d_feats, 'b (ncam p) f -> b ncam p f', ncam=ncam).mean(dim=2)
        per_img_pos = einops.rearrange(pcd, 'b (ncam p) c -> b ncam p c', ncam=ncam).mean(dim=2)

        fps_scene_feats = torch.cat([fps_scene_feats, per_img_feats], dim=1)
        fps_scene_pos = torch.cat([fps_scene_pos, per_img_pos], dim=1)

        # fps_cam_ids: (B, M) — camera index for each fps token
        # per-image token for camera c is at fps_scene_feats[:, M + c]

        return (
            rgb3d_feats, pcd,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos,
            fps_cam_ids
        )

    def encode_proprio(self, proprio, context_feats, context_pos):
        """
        Compute proprioception features.

        Args:
            - proprio: (B, nhist, 3+)
            - context_feats: (B, npt, C)
            - context_pos: (B, npt, 3)

        Returns:
            - gripper_feats: (B, nhist, F)
        """
        return None

    def encode_clip(self, rgb3d, rgb2d, pcd, text):
        """
        Compute visual features/pos embeddings.

        Args:
            - rgb3d: (B, ncam3d, 3, H, W), rgb obs of 3D cameras
            - rgb2d: (B, ncam2d, 3, H, W), rgb obs of 2D cameras
            - pcd: (B, ncam3d, 3, H, W) or None
            - text: [str] of len=B, text instruction

        Returns:
            - rgb3d_feats: (B, Np, F)
            - rgb2d_feats: (B, ncam2d, F)
            - pcd: (B, Np, 3)
            - instr_feats: (B, L, F)
        """
        return None, None, None, None

    def encode_dino(self, rgb3d, rgb2d, pcd, text):
        """Stub — implemented in subclass."""
        return None, None, None, None

    def run_dps(self, features, pos, cam_ids=None):
        # features (B, Np, F), pos (B, Np, 3), cam_ids (B, Np) optional
        # outputs of analogous shape, with smaller Np
        if self.subsampling_factor == 1:
            return features, pos, cam_ids

        bs, npts, ch = features.shape
        sampled_inds = density_based_sampler(features, self.subsampling_factor)

        # Sample features
        expanded_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, ch)  # B M F
        sampled_features = torch.gather(features, 1, expanded_inds)

        # Sample cam_ids if provided
        sampled_cam_ids = torch.gather(cam_ids, 1, sampled_inds) if cam_ids is not None else None

        # If positions are None, return
        if pos is None:
            return sampled_features, None, sampled_cam_ids

        # Else sample positions
        expanded_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, 3)  # B M 3
        sampled_pos = torch.gather(pos, 1, expanded_inds)
        return sampled_features, sampled_pos, sampled_cam_ids


@torch.no_grad()
def density_based_sampler(features, subsample_factor, k=8):
    """
    Args:
        features: Tensor of shape (B, N, C)
        subsample_factor: downsampling factor, e.g., 4 keeps 25% of the points
        k: number of neighbors to compute local density (default: 8)

    Returns:
        sampled_inds: LongTensor (B, N//factor) with sampled point indices
    """
    B, N, C = features.shape
    # (B, N, N) pairwise distances in feature space
    dists = torch.cdist(features, features, p=2)  # L2 distance

    # Get average distance to k nearest neighbors (as inverse density estimate)
    knn_dists, _ = dists.topk(k=k, dim=-1, largest=False)
    density = knn_dists.mean(dim=-1)  # (B, N), higher = more sparse

    # Choose top M points with highest avg distance (i.e. lowest density)
    M = int(N // subsample_factor)
    sampled_inds = density.topk(M, dim=-1, largest=True).indices  # (B, M)

    return sampled_inds
