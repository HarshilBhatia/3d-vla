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
        self._finetune_backbone = finetune_backbone
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
        if not finetune_backbone:
            # Frozen backbones should not update BatchNorm/other running stats.
            self.backbone.eval()

        # Attention from vision to language
        if backbone in ('clip', 'siglip2', 'dino'):
            self.vl_attention = AttentionModule(
                num_layers=num_vis_instr_attn_layers, d_model=embedding_dim,
                dim_fw=4 * embedding_dim, n_heads=num_attn_heads, pre_norm=False
            )

    def train(self, mode=True):
        super().train(mode)
        if not self._finetune_backbone and self.backbone is not None:
            # Keep frozen vision backbone in eval mode even when parent model trains.
            self.backbone.eval()
        return self

    def forward(self, rgb3d, rgb2d, pcd, instruction, proprio, stopgrad_k=0):
        """
        Encode different modalities, independent of denoising step.

        Args:
            - rgb3d: (B, ncam3d, 3, H, W) or (B, nhist, ncam3d, 3, H, W)
            - pcd:   same spatial shape as rgb3d
            - proprio: (B, nhist, 3+6+X)

        Returns:
            - rgb3d_feats: (B, Np, F) or (B, nhist, Np, F) when nhist > 1
            - pcd_out: (B, Np, 3) or (B, nhist, Np, 3)
            - fps_scene_feats/pos: always built from the CURRENT (latest) frame
        """
        vl_enc_fn = {
            'clip': self.encode_clip,
            'siglip2': self.encode_siglip2,
            'dino': self.encode_dino,
        }[self._backbone_name]
        # Compute scene features/positional embeddings, language embeddings
        rgb3d_feats, rgb2d_feats, pcd_out, instr_feats = vl_enc_fn(
            rgb3d, rgb2d, pcd, instruction
        )
        rgb2d_pos = None

        # When nhist > 1, FPS and per-image tokens use only the latest frame
        if rgb3d_feats.ndim == 4:  # (B, nhist, ncam*P, F)
            rgb3d_feats_curr = rgb3d_feats[:, -1]  # (B, ncam*P, F)
            pcd_curr = pcd_out[:, -1]              # (B, ncam*P, 3)
        else:
            rgb3d_feats_curr = rgb3d_feats
            pcd_curr = pcd_out

        # Use the current end-effector position as language 'position'
        instr_pos = proprio[:, -1:, :3].repeat(1, instr_feats.size(1), 1)

        # Encode proprioception using current frame's scene context
        proprio_feats = self.encode_proprio(
            proprio, rgb3d_feats_curr, pcd_curr, stopgrad_k=stopgrad_k
        )

        # Build (B, Np) camera-index tensor from current frame
        ncam = rgb3d.shape[-4]  # works for both 5D (B, ncam, ...) and 6D (B, nhist, ncam, ...)
        Np = rgb3d_feats_curr.shape[1]
        cam_ids_full = (
            torch.arange(ncam, device=rgb3d_feats_curr.device)
            .repeat_interleave(Np // ncam)
            .unsqueeze(0).expand(rgb3d_feats_curr.shape[0], -1)
        )

        # Point subsampling from current frame
        fps_scene_feats, fps_scene_pos, fps_cam_ids = self.run_dps(
            rgb3d_feats_curr, pcd_curr, cam_ids_full
        )

        # Per-image average tokens from current frame
        per_img_feats = einops.rearrange(
            rgb3d_feats_curr, 'b (ncam p) f -> b ncam p f', ncam=ncam
        ).mean(dim=2)
        per_img_pos = einops.rearrange(
            pcd_curr, 'b (ncam p) c -> b ncam p c', ncam=ncam
        ).mean(dim=2)

        fps_scene_feats = torch.cat([fps_scene_feats, per_img_feats], dim=1)
        fps_scene_pos = torch.cat([fps_scene_pos, per_img_pos], dim=1)

        return (
            rgb3d_feats, pcd_out,
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
