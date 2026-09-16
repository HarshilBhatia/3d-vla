import math

import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RotaryPositionEncoding(nn.Module):
    def __init__(self, feature_dim, pe_type='Rotary1D'):
        super().__init__()

        self.feature_dim = feature_dim
        self.pe_type = pe_type

    @staticmethod
    def embed_rotary(x, cos, sin):
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    def forward(self, x_position):
        bsize, npoint = x_position.shape
        div_term = torch.exp(
            torch.arange(0, self.feature_dim, 2, dtype=torch.float, device=x_position.device)
            * (-math.log(10000.0) / (self.feature_dim)))
        div_term = div_term.view(1, 1, -1) # [1, 1, d]

        sinx = torch.sin(x_position * div_term)  # [B, N, d]
        cosx = torch.cos(x_position * div_term)

        sin_pos, cos_pos = map(
            lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
            [sinx, cosx]
        )
        position_code = torch.stack([cos_pos, sin_pos] , dim=-1)

        # Always detach for base RotaryPositionEncoding (not used for pcd)
        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


class PositionEmbeddingLearnedMLP(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, dim=3, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Linear(dim, num_pos_feats),
            nn.LayerNorm(num_pos_feats),
            nn.ReLU(),
            nn.Linear(num_pos_feats, num_pos_feats))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, F, N)."""
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding



def _apply_delta_M(feat, delta_M, ncam=None):
    """Apply delta_M to sin/cos feature stack [B, N, d//6, 6].

    Dispatch:
      (B, N, 6, 6)           — per-token 6×6
      (B, 6, 6)              — broadcast 6×6
      (B, ncam, D, D)+ncam   — grouped per-camera D×D (no per-token expand)
      (B, N, D, D)           — per-token D×D (fallback)
      (B, D, D)              — broadcast D×D
    """
    bsz, np2, nb, _ = feat.shape
    if delta_M.ndim == 4 and delta_M.shape[-1] == 6:        # (B, N, 6, 6)
        return torch.einsum('bnci,bnji->bncj', feat, delta_M)
    elif delta_M.ndim == 3 and delta_M.shape[-1] == 6:      # (B, 6, 6)
        return torch.einsum('bnci,bji->bncj', feat, delta_M)
    elif delta_M.ndim == 4 and ncam is not None:             # (B, ncam, D, D) grouped
        P = np2 // ncam
        # (B, ncam, P, D) @ (B, ncam, D, D).T  — direct cuBLAS batched GEMM, no expand
        out = (feat.reshape(bsz, ncam, P, nb * 6) @ delta_M.transpose(-1, -2))
        return out.reshape(bsz, np2, nb, 6)
    elif delta_M.ndim == 4:                                  # (B, N, D, D) per-token
        return torch.einsum('bni,bnji->bnj',
                            feat.reshape(bsz, np2, -1),
                            delta_M).reshape(bsz, np2, nb, 6)
    else:                                                    # (B, D, D) broadcast
        return torch.einsum('bni,bji->bnj',
                            feat.reshape(bsz, np2, -1),
                            delta_M).reshape(bsz, np2, nb, 6)


class RotaryPositionEncoding3D(RotaryPositionEncoding):
    # NOTE: adjust inheritance to match your actual parent class

    def __init__(self, feature_dim, pe_type='Rotary3D'):
        """
        Args:
            feature_dim: Dimension of the position encoding features
            pe_type: Type of position encoding (default: 'Rotary3D')
        """
        super().__init__(feature_dim, pe_type)
        self.feature_dim = feature_dim

    def forward(self, XYZ, allow_grad=False, delta_M=None, ncam=None):
        '''
        @param XYZ: [B,N,3]
        @param allow_grad: whether to allow gradients to flow through
        @param delta_M: optional (B, 6, 6) to mix sin/cos features before view/stack (predict delta M from cam_token)
        @return: position_code [B, N, feature_dim, 2]
        '''
        bsize, npoint, _ = XYZ.shape
        x_position, y_position, z_position = XYZ[..., 0:1], XYZ[..., 1:2], XYZ[..., 2:3]
        dx = dy = self.feature_dim // 3
        if dx % 2 == 1:
            dx -= 1
            dy -= 1
        dz = self.feature_dim - dx - dy
        div_term_x = torch.exp(
            torch.arange(0, dx, 2, dtype=torch.float, device=XYZ.device)
            * (-math.log(10000.0) / dx)
        ).view(1, 1, -1)  # [1, 1, d//6]
        div_term_y = torch.exp(
            torch.arange(0, dy, 2, dtype=torch.float, device=XYZ.device)
            * (-math.log(10000.0) / dy)
        ).view(1, 1, -1)  # [1, 1, d//6]
        div_term_z = torch.exp(
            torch.arange(0, dz, 2, dtype=torch.float, device=XYZ.device)
            * (-math.log(10000.0) / dz)
        ).view(1, 1, -1)  # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term_x)  # [B, N, d//6]
        cosx = torch.cos(x_position * div_term_x)
        siny = torch.sin(y_position * div_term_y)
        cosy = torch.cos(y_position * div_term_y)
        sinz = torch.sin(z_position * div_term_z)
        cosz = torch.cos(z_position * div_term_z)

        # Optional: mix sin/cos with delta_M (from cam_token), before view/stack
        if delta_M is not None:
            feat = torch.stack([cosx, cosy, cosz, sinx, siny, sinz], dim=-1)  # [B, N, d//6, 6]
            feat = _apply_delta_M(feat, delta_M, ncam)
            cosx, cosy, cosz = feat[..., 0], feat[..., 1], feat[..., 2]
            sinx, siny, sinz = feat[..., 3], feat[..., 4], feat[..., 5]

        sinx, cosx, siny, cosy, sinz, cosz = map(
            lambda feat: torch.stack([feat, feat], -1).view(bsize, npoint, -1),
            [sinx, cosx, siny, cosy, sinz, cosz]
        )

        position_code = torch.stack([
            torch.cat([cosx, cosy, cosz], dim=-1),  # cos_pos
            torch.cat([sinx, siny, sinz], dim=-1)   # sin_pos
        ], dim=-1)

        # Only detach if gradients are not wanted; otherwise grad flows through XYZ.
        if not allow_grad:
            position_code = position_code.detach()

        return position_code

    def _compute_sincos_base(self, XYZ):
        """Compute raw sin/cos stack for XYZ without delta_M mixing.

        Returns [B, N, d//6, 6] = [cosx, cosy, cosz, sinx, siny, sinz] stacked on last dim.
        Always detached — use forward() when gradients through XYZ are needed.
        """
        bsize, npoint, _ = XYZ.shape
        x_position, y_position, z_position = XYZ[..., 0:1], XYZ[..., 1:2], XYZ[..., 2:3]
        dx = dy = self.feature_dim // 3
        if dx % 2 == 1:
            dx -= 1
            dy -= 1
        dz = self.feature_dim - dx - dy
        div_term_x = torch.exp(
            torch.arange(0, dx, 2, dtype=torch.float, device=XYZ.device)
            * (-math.log(10000.0) / dx)
        ).view(1, 1, -1)
        div_term_y = torch.exp(
            torch.arange(0, dy, 2, dtype=torch.float, device=XYZ.device)
            * (-math.log(10000.0) / dy)
        ).view(1, 1, -1)
        div_term_z = torch.exp(
            torch.arange(0, dz, 2, dtype=torch.float, device=XYZ.device)
            * (-math.log(10000.0) / dz)
        ).view(1, 1, -1)

        sinx = torch.sin(x_position * div_term_x)
        cosx = torch.cos(x_position * div_term_x)
        siny = torch.sin(y_position * div_term_y)
        cosy = torch.cos(y_position * div_term_y)
        sinz = torch.sin(z_position * div_term_z)
        cosz = torch.cos(z_position * div_term_z)

        base = torch.stack([cosx, cosy, cosz, sinx, siny, sinz], dim=-1)  # [B, N, d//6, 6]
        return base.detach()

    def _finalize_from_base(self, base_feat, delta_M=None, ncam=None):
        """Apply optional delta_M to pre-computed sin/cos base and return position_code [B,N,C,2].

        Args:
            base_feat: [B, N, d//6, 6] — output of _compute_sincos_base
            delta_M: optional matrix to mix sin/cos features
            ncam: if provided and delta_M is (B, ncam, D, D), use grouped per-camera matmul

        Returns:
            position_code: [B, N, C, 2]
        """
        bsize, npoint = base_feat.shape[:2]

        if delta_M is not None:
            feat = _apply_delta_M(base_feat, delta_M, ncam)
            cosx, cosy, cosz = feat[..., 0], feat[..., 1], feat[..., 2]
            sinx, siny, sinz = feat[..., 3], feat[..., 4], feat[..., 5]
        else:
            cosx, cosy, cosz = base_feat[..., 0], base_feat[..., 1], base_feat[..., 2]
            sinx, siny, sinz = base_feat[..., 3], base_feat[..., 4], base_feat[..., 5]

        sinx, cosx, siny, cosy, sinz, cosz = map(
            lambda feat: torch.stack([feat, feat], -1).view(bsize, npoint, -1),
            [sinx, cosx, siny, cosy, sinz, cosz]
        )

        position_code = torch.stack([
            torch.cat([cosx, cosy, cosz], dim=-1),
            torch.cat([sinx, siny, sinz], dim=-1)
        ], dim=-1)

        return position_code