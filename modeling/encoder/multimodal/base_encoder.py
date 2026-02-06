import torch
from torch import nn

from ...utils.layers import AttentionModule
from ..vision import fetch_visual_encoders
from ..text import fetch_text_encoders


class Encoder(nn.Module):

    def __init__(self,
                 backbone="clip",
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

        # Instruction encoder
        self.text_encoder, _dim = fetch_text_encoders(backbone)
        if self.text_encoder is not None:  # is None when using a VLM
            for p in self.text_encoder.parameters():
                p.requires_grad = finetune_text_encoder
            self.instruction_encoder = nn.Linear(_dim, embedding_dim)

        # Scene encoder
        self.backbone, self.normalize = fetch_visual_encoders(backbone)
        for p in self.backbone.parameters():
            p.requires_grad = finetune_backbone

        # Attention from vision to language
        if backbone == 'clip':
            self.vl_attention = AttentionModule(
                num_layers=num_vis_instr_attn_layers, d_model=embedding_dim,
                dim_fw=4 * embedding_dim, n_heads=num_attn_heads, pre_norm=False
            )

    def forward(self, rgb3d, rgb2d, pcd, instruction, proprio):
        """
        Encode different modalities, independent of denoising step.

        Args:
            - rgb3d: (B, ncam3d, 3, H, W)
            - rgb2d: (B, ncam2d, 3, H, W)
            - pcd: (B, ncam3d, 3, H, W)
            - instruction: (B, nt), tokens
            - proprio: (B, nhist, 3+6+X)

        Returns:
            - rgb3d_feats: (B, N, F)
            - pcd: (B, N, 3)
            - rgb2d_feats: (B, N2d, F)
            - rgb2d_pos: (B, N2d, 3)
            - instr_feats: (B, L, F)
            - instr_pos: (B, L, 3)
            - proprio_feats: (B, nhist, F)
            - fps_scene_feats: (B, n, F), n < N
            - fps_scene_pos: (B, n, 3), n < N
            - scene_dbs_density: (B, N) or None, DBS sparsity score per point
        """
        vl_enc_fn = {
            'clip': self.encode_clip,
        }[self._backbone_name]
        # Compute scene features/positional embeddings, language embeddings
        rgb3d_feats, rgb2d_feats, pcd, instr_feats = vl_enc_fn(
            rgb3d, rgb2d, pcd, instruction
        )
        rgb2d_pos = None

        # Use the current end-effector position as language 'position'
        instr_pos = proprio[:, -1:, :3].repeat(1, instr_feats.size(1), 1)

        # Encode proprioception
        proprio_feats = self.encode_proprio(proprio, rgb3d_feats, pcd)

        # Point subsampling based on scene features
        fps_scene_feats, fps_scene_pos, scene_dbs_density = self.run_dps(
            rgb3d_feats, pcd
        )

        return (
            rgb3d_feats, pcd,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos,
            scene_dbs_density
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

    def run_dps(self, features, pos):
        # features (B, Np, F)
        # context_pos (B, Np, 3)
        # outputs of analogous shape, with smaller Np
        if self.subsampling_factor == 1:
            return features, pos, None

        bs, npts, ch = features.shape
        scene_dbs_density = density_based_scores(features)

        # Choose top M points with highest avg distance (i.e. lowest density)
        M = int(npts // self.subsampling_factor)
        sampled_inds = scene_dbs_density.topk(M, dim=-1, largest=True).indices

        # Sample features
        expanded_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, ch)  # B Np F
        sampled_features = torch.gather(features, 1, expanded_inds)

        # If positions are None, return
        if pos is None:
            return sampled_features, None, scene_dbs_density

        # Else sample positions
        expanded_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, 3)  # B Np 3
        sampled_pos = torch.gather(pos, 1, expanded_inds)
        return sampled_features, sampled_pos, scene_dbs_density


@torch.no_grad()
def density_based_scores(features, k=8):
    """
    Args:
        features: Tensor of shape (B, N, C)
        k: number of neighbors to compute local density (default: 8)

    Returns:
        density: FloatTensor (B, N), higher = more sparse in feature space
    """
    B, N, C = features.shape
    if N == 0:
        return features.new_zeros((B, 0))
    # (B, N, N) pairwise distances in feature space
    dists = torch.cdist(features, features, p=2)  # L2 distance

    # Get average distance to k nearest neighbors (as inverse density estimate)
    k_eff = min(int(k), int(N))
    knn_dists, _ = dists.topk(k=k_eff, dim=-1, largest=False)
    density = knn_dists.mean(dim=-1)  # (B, N), higher = more sparse
    return density


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
    B, N, _ = features.shape
    if N == 0:
        return torch.empty((B, 0), dtype=torch.long, device=features.device)
    density = density_based_scores(features, k=k)
    M = int(N // subsample_factor)
    return density.topk(M, dim=-1, largest=True).indices  # (B, M)


@torch.no_grad()
def adaptive_trajectory_sample_inds(
    scene_pos,
    scene_dbs_density,
    subsample_factor,
    traj_xyz,
    sigma=0.03,
    beta=1.0,
    eps=1e-6
):
    """
    Adaptive Trajectory-Centric Sampling indices.

    Combines DBS sparsity score (feature-space inverse density) with a spatial
    importance weight around the current trajectory estimate tau_i:

        W_p = exp( - min_t ||p - tau_i[t]||^2 / sigma^2 )

    Final sampling score:
        score = norm(DBS) + beta * W_p

    Args:
        scene_pos: (B, N, 3) raw 3D locations for each visual token
        scene_dbs_density: (B, N) DBS sparsity score per token (higher = sparser)
        subsample_factor: int, downsampling factor (e.g., 4 keeps 25%)
        traj_xyz: (B, T, 3) current trajectory estimate points
        sigma: float, spatial falloff (in same units as scene_pos)
        beta: float, weight for the trajectory-centric term

    Returns:
        sampled_inds: (B, M) LongTensor indices, where M = N//subsample_factor
    """
    B, N, _ = scene_pos.shape
    if N == 0:
        return torch.empty((B, 0), dtype=torch.long, device=scene_pos.device)
    M = int(N // int(subsample_factor))

    # If missing trajectory / density, fall back to DBS.
    if (
        traj_xyz is None
        or scene_dbs_density is None
        or traj_xyz.numel() == 0
        or sigma is None
        or float(sigma) <= 0.0
        or float(beta) <= 0.0
    ):
        return scene_dbs_density.topk(M, dim=-1, largest=True).indices

    # Normalize DBS density into [0, 1] per batch element.
    dbs = scene_dbs_density
    dbs_min = dbs.min(dim=-1, keepdim=True).values
    dbs_max = dbs.max(dim=-1, keepdim=True).values
    dbs_norm = (dbs - dbs_min) / (dbs_max - dbs_min + eps)

    # Compute min squared distance from each scene point to any trajectory point.
    # (B, N, T, 3) -> (B, N, T) -> (B, N)
    diff = scene_pos.unsqueeze(2) - traj_xyz.unsqueeze(1)
    min_d2 = (diff * diff).sum(dim=-1).min(dim=-1).values
    w = torch.exp(-min_d2 / (float(sigma) * float(sigma) + eps))

    score = dbs_norm + float(beta) * w
    return score.topk(M, dim=-1, largest=True).indices
