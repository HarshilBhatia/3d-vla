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



def _interleave_xyz_cos_sin(cosx, cosy, cosz, sinx, siny, sinz):
    """Build (x1,y1,z1, x2,y2,z2, ...) layout from per-axis cos/sin. Each input [B, N, axis_len]."""
    B, N = cosx.shape[:2]
    dx, dy, dz = cosx.shape[-1], cosy.shape[-1], cosz.shape[-1]
    n_bins = min(dx, dy, dz) // 2  # number of (x,y,z) triplets of RoPE pairs
    # Interleaved block: for each bin i take 2 dims from x, y, z
    cos_parts, sin_parts = [], []
    for i in range(n_bins):
        cos_parts.extend([cosx[:, :, 2 * i : 2 * i + 2], cosy[:, :, 2 * i : 2 * i + 2], cosz[:, :, 2 * i : 2 * i + 2]])
        sin_parts.extend([sinx[:, :, 2 * i : 2 * i + 2], siny[:, :, 2 * i : 2 * i + 2], sinz[:, :, 2 * i : 2 * i + 2]])
    cos_inter = torch.cat(cos_parts, dim=-1)  # [B, N, 6*n_bins]
    sin_inter = torch.cat(sin_parts, dim=-1)
    # Remainder: rest of x, then y, then z
    rem_x = dx - 2 * n_bins
    rem_y = dy - 2 * n_bins
    rem_z = dz - 2 * n_bins
    if rem_x > 0 or rem_y > 0 or rem_z > 0:
        cos_rem = torch.cat(
            [
                cosx[:, :, 2 * n_bins :],
                cosy[:, :, 2 * n_bins :],
                cosz[:, :, 2 * n_bins :],
            ],
            dim=-1,
        )
        sin_rem = torch.cat(
            [
                sinx[:, :, 2 * n_bins :],
                siny[:, :, 2 * n_bins :],
                sinz[:, :, 2 * n_bins :],
            ],
            dim=-1,
        )
        cos_inter = torch.cat([cos_inter, cos_rem], dim=-1)
        sin_inter = torch.cat([sin_inter, sin_rem], dim=-1)
    return cos_inter, sin_inter, n_bins, dx, dy, dz


class RoPE3DAdamFunction(torch.autograd.Function):
    """
    Custom autograd for 3D RoPE that normalizes per-bin gradients
    using a per-bin second moment (RMSProp-style) before summing.

    Forward: identical to standard 3D RoPE.
    Backward: decomposes grad into per-bin contributions analytically,
              normalizes each by its running second moment v[i],
              then sums back to grad_XYZ.
    """

    @staticmethod
    def forward(ctx, XYZ, div_term_x, div_term_y, div_term_z, adam_v):
        bsize, npoint, _ = XYZ.shape
        x_position, y_position, z_position = XYZ[..., 0:1], XYZ[..., 1:2], XYZ[..., 2:3]

        # Compute sin/cos per axis (pre-repeat versions saved for backward)
        sinx = torch.sin(x_position * div_term_x)  # [B, N, dx//2]
        cosx = torch.cos(x_position * div_term_x)
        siny = torch.sin(y_position * div_term_y)
        cosy = torch.cos(y_position * div_term_y)
        sinz = torch.sin(z_position * div_term_z)
        cosz = torch.cos(z_position * div_term_z)

        ctx.save_for_backward(sinx, cosx, siny, cosy, sinz, cosz,
                              div_term_x, div_term_y, div_term_z)
        ctx.adam_v = adam_v  # mutable reference to module buffer, updated in-place
        ctx.dx = div_term_x.shape[-1] * 2
        ctx.dy = div_term_y.shape[-1] * 2

        # Repeat each bin twice and build position_code (interleaved: x1,y1,z1, x2,y2,z2, ...)
        sinx, cosx, siny, cosy, sinz, cosz = map(
            lambda feat: torch.stack([feat, feat], -1).view(bsize, npoint, -1),
            [sinx, cosx, siny, cosy, sinz, cosz]
        )
        cos_inter, sin_inter, n_bins, dx, dy, dz = _interleave_xyz_cos_sin(
            cosx, cosy, cosz, sinx, siny, sinz
        )
        position_code = torch.stack([cos_inter, sin_inter], dim=-1)
        ctx.n_bins = n_bins
        ctx.dx, ctx.dy, ctx.dz = dx, dy, dz

        return position_code

    @staticmethod
    def backward(ctx, grad_output):
        sinx, cosx, siny, cosy, sinz, cosz, div_term_x, div_term_y, div_term_z = ctx.saved_tensors
        adam_v = ctx.adam_v
        n_bins = ctx.n_bins
        dx, dy, dz = ctx.dx, ctx.dy, ctx.dz
        beta2, eps = 0.999, 1e-8

        B, N = grad_output.shape[:2]
        nbx = div_term_x.shape[-1]   # num bins x  (= dx//2)
        nby = div_term_y.shape[-1]
        nbz = div_term_z.shape[-1]

        def sum_repeat(grad_slice, num_bins):
            """[B, N, 2*num_bins] -> [B, N, num_bins] by summing the two copies."""
            return grad_slice.view(B, N, num_bins, 2).sum(-1)

        # De-interleave: x bins at 6*i, 6*i+1; y at 6*i+2, 6*i+3; z at 6*i+4, 6*i+5
        gx_cos_parts = [grad_output[:, :, 6 * i : 6 * i + 2, 0] for i in range(n_bins)]
        gx_sin_parts = [grad_output[:, :, 6 * i : 6 * i + 2, 1] for i in range(n_bins)]
        gy_cos_parts = [grad_output[:, :, 6 * i + 2 : 6 * i + 4, 0] for i in range(n_bins)]
        gy_sin_parts = [grad_output[:, :, 6 * i + 2 : 6 * i + 4, 1] for i in range(n_bins)]
        gz_cos_parts = [grad_output[:, :, 6 * i + 4 : 6 * i + 6, 0] for i in range(n_bins)]
        gz_sin_parts = [grad_output[:, :, 6 * i + 4 : 6 * i + 6, 1] for i in range(n_bins)]
        gx_cos = sum_repeat(torch.cat(gx_cos_parts, dim=-1), n_bins)
        gx_sin = sum_repeat(torch.cat(gx_sin_parts, dim=-1), n_bins)
        gy_cos = sum_repeat(torch.cat(gy_cos_parts, dim=-1), n_bins)
        gy_sin = sum_repeat(torch.cat(gy_sin_parts, dim=-1), n_bins)
        gz_cos = sum_repeat(torch.cat(gz_cos_parts, dim=-1), n_bins)
        gz_sin = sum_repeat(torch.cat(gz_sin_parts, dim=-1), n_bins)
        # Remainder: 6*n_bins : 6*n_bins+dx-2*n_bins = x_rem, then y_rem, then z_rem
        base = 6 * n_bins
        rem_x, rem_y, rem_z = dx - 2 * n_bins, dy - 2 * n_bins, dz - 2 * n_bins
        if rem_x > 0:
            gx_cos = torch.cat([gx_cos, sum_repeat(grad_output[:, :, base : base + rem_x, 0], rem_x // 2)], dim=-1)
            gx_sin = torch.cat([gx_sin, sum_repeat(grad_output[:, :, base : base + rem_x, 1], rem_x // 2)], dim=-1)
            base += rem_x
        if rem_y > 0:
            gy_cos = torch.cat([gy_cos, sum_repeat(grad_output[:, :, base : base + rem_y, 0], rem_y // 2)], dim=-1)
            gy_sin = torch.cat([gy_sin, sum_repeat(grad_output[:, :, base : base + rem_y, 1], rem_y // 2)], dim=-1)
            base += rem_y
        if rem_z > 0:
            gz_cos = torch.cat([gz_cos, sum_repeat(grad_output[:, :, base : base + rem_z, 0], rem_z // 2)], dim=-1)
            gz_sin = torch.cat([gz_sin, sum_repeat(grad_output[:, :, base : base + rem_z, 1], rem_z // 2)], dim=-1)

        # Per-bin chain rule:
        #   d cos(pos * theta) / d pos = -theta * sin(pos * theta)
        #   d sin(pos * theta) / d pos =  theta * cos(pos * theta)
        grad_x = gx_cos * (-div_term_x * sinx) + gx_sin * (div_term_x * cosx)  # [B, N, nbx]
        grad_y = gy_cos * (-div_term_y * siny) + gy_sin * (div_term_y * cosy)  # [B, N, nby]
        grad_z = gz_cos * (-div_term_z * sinz) + gz_sin * (div_term_z * cosz)  # [B, N, nbz]

        # --- Per-bin Adam (v only) ---
        # Average over B, N to get a scalar per bin for the moment update
        per_bin_mean = torch.cat([
            grad_x.mean((0, 1)),   # [nbx]
            grad_y.mean((0, 1)),   # [nby]
            grad_z.mean((0, 1)),   # [nbz]
        ])  # [total_bins]

        # Update second moment in-place on the module buffer
        adam_v.mul_(beta2).add_(per_bin_mean ** 2, alpha=1 - beta2)

        # Normalize each bin's gradient by its second moment
        v_x = adam_v[:nbx]
        v_y = adam_v[nbx:nbx + nby]
        v_z = adam_v[nbx + nby:]

        grad_x = grad_x / (v_x.sqrt() + eps)
        grad_y = grad_y / (v_y.sqrt() + eps)
        grad_z = grad_z / (v_z.sqrt() + eps)

        # Sum across bins -> grad per axis -> grad_XYZ
        grad_XYZ = torch.cat([
            grad_x.sum(-1, keepdim=True),  # [B, N, 1]
            grad_y.sum(-1, keepdim=True),
            grad_z.sum(-1, keepdim=True),
        ], dim=-1)  # [B, N, 3]

        # Returns: grad for XYZ, None for div_terms and adam_v (not learnable)
        return grad_XYZ, None, None, None, None


class RoPE3DStopGradFunction(torch.autograd.Function):
    """
    Custom autograd for 3D RoPE with stopgrad behavior.
    
    Forward: identical to standard 3D RoPE.
    Backward: zeros out gradients for the first K bins.
    """

    @staticmethod
    def forward(ctx, XYZ, div_term_x, div_term_y, div_term_z, stopgrad_k):
        bsize, npoint, _ = XYZ.shape
        x_position, y_position, z_position = XYZ[..., 0:1], XYZ[..., 1:2], XYZ[..., 2:3]

        # Compute sin/cos per axis (pre-repeat versions saved for backward)
        sinx = torch.sin(x_position * div_term_x)  # [B, N, dx//2]
        cosx = torch.cos(x_position * div_term_x)
        siny = torch.sin(y_position * div_term_y)
        cosy = torch.cos(y_position * div_term_y)
        sinz = torch.sin(z_position * div_term_z)
        cosz = torch.cos(z_position * div_term_z)

        ctx.save_for_backward(sinx, cosx, siny, cosy, sinz, cosz,
                              div_term_x, div_term_y, div_term_z)
        ctx.stopgrad_k = stopgrad_k

        # Repeat each bin twice and build position_code (interleaved: x1,y1,z1, x2,y2,z2, ...)
        sinx, cosx, siny, cosy, sinz, cosz = map(
            lambda feat: torch.stack([feat, feat], -1).view(bsize, npoint, -1),
            [sinx, cosx, siny, cosy, sinz, cosz]
        )
        cos_inter, sin_inter, n_bins, dx, dy, dz = _interleave_xyz_cos_sin(
            cosx, cosy, cosz, sinx, siny, sinz
        )
        position_code = torch.stack([cos_inter, sin_inter], dim=-1)
        ctx.n_bins = n_bins
        ctx.dx, ctx.dy, ctx.dz = dx, dy, dz

        return position_code

    @staticmethod
    def backward(ctx, grad_output):
        sinx, cosx, siny, cosy, sinz, cosz, div_term_x, div_term_y, div_term_z = ctx.saved_tensors
        stopgrad_k = ctx.stopgrad_k
        n_bins = ctx.n_bins
        dx, dy, dz = ctx.dx, ctx.dy, ctx.dz

        B, N = grad_output.shape[:2]
        nbx = div_term_x.shape[-1]   # num bins x  (= dx//2)
        nby = div_term_y.shape[-1]
        nbz = div_term_z.shape[-1]

        def sum_repeat(grad_slice, num_bins):
            """[B, N, 2*num_bins] -> [B, N, num_bins] by summing the two copies."""
            return grad_slice.view(B, N, num_bins, 2).sum(-1)

        # De-interleave: x at 6*i, 6*i+1; y at 6*i+2, 6*i+3; z at 6*i+4, 6*i+5
        gx_cos_parts = [grad_output[:, :, 6 * i : 6 * i + 2, 0] for i in range(n_bins)]
        gx_sin_parts = [grad_output[:, :, 6 * i : 6 * i + 2, 1] for i in range(n_bins)]
        gy_cos_parts = [grad_output[:, :, 6 * i + 2 : 6 * i + 4, 0] for i in range(n_bins)]
        gy_sin_parts = [grad_output[:, :, 6 * i + 2 : 6 * i + 4, 1] for i in range(n_bins)]
        gz_cos_parts = [grad_output[:, :, 6 * i + 4 : 6 * i + 6, 0] for i in range(n_bins)]
        gz_sin_parts = [grad_output[:, :, 6 * i + 4 : 6 * i + 6, 1] for i in range(n_bins)]
        gx_cos = sum_repeat(torch.cat(gx_cos_parts, dim=-1), n_bins)
        gx_sin = sum_repeat(torch.cat(gx_sin_parts, dim=-1), n_bins)
        gy_cos = sum_repeat(torch.cat(gy_cos_parts, dim=-1), n_bins)
        gy_sin = sum_repeat(torch.cat(gy_sin_parts, dim=-1), n_bins)
        gz_cos = sum_repeat(torch.cat(gz_cos_parts, dim=-1), n_bins)
        gz_sin = sum_repeat(torch.cat(gz_sin_parts, dim=-1), n_bins)
        base = 6 * n_bins
        rem_x, rem_y, rem_z = dx - 2 * n_bins, dy - 2 * n_bins, dz - 2 * n_bins
        if rem_x > 0:
            gx_cos = torch.cat([gx_cos, sum_repeat(grad_output[:, :, base : base + rem_x, 0], rem_x // 2)], dim=-1)
            gx_sin = torch.cat([gx_sin, sum_repeat(grad_output[:, :, base : base + rem_x, 1], rem_x // 2)], dim=-1)
            base += rem_x
        if rem_y > 0:
            gy_cos = torch.cat([gy_cos, sum_repeat(grad_output[:, :, base : base + rem_y, 0], rem_y // 2)], dim=-1)
            gy_sin = torch.cat([gy_sin, sum_repeat(grad_output[:, :, base : base + rem_y, 1], rem_y // 2)], dim=-1)
            base += rem_y
        if rem_z > 0:
            gz_cos = torch.cat([gz_cos, sum_repeat(grad_output[:, :, base : base + rem_z, 0], rem_z // 2)], dim=-1)
            gz_sin = torch.cat([gz_sin, sum_repeat(grad_output[:, :, base : base + rem_z, 1], rem_z // 2)], dim=-1)

        # Per-bin chain rule:
        #   d cos(pos * theta) / d pos = -theta * sin(pos * theta)
        #   d sin(pos * theta) / d pos =  theta * cos(pos * theta)
        grad_x = gx_cos * (-div_term_x * sinx) + gx_sin * (div_term_x * cosx)  # [B, N, nbx]
        grad_y = gy_cos * (-div_term_y * siny) + gy_sin * (div_term_y * cosy)  # [B, N, nby]
        grad_z = gz_cos * (-div_term_z * sinz) + gz_sin * (div_term_z * cosz)  # [B, N, nbz]

        # --- Apply stopgrad masking: zero out first K bins ---
        if stopgrad_k > 0:
            total_bins = nbx + nby + nbz
            k = max(0, min(stopgrad_k, total_bins))  # Clamp to valid range
            
            # Determine how many bins to mask from each axis
            k_remaining = k
            
            # Mask x bins first
            if k_remaining > 0:
                k_x = min(k_remaining, nbx)
                grad_x[:, :, :k_x] = 0
                k_remaining -= k_x
            
            # Then mask y bins
            if k_remaining > 0:
                k_y = min(k_remaining, nby)
                grad_y[:, :, :k_y] = 0
                k_remaining -= k_y
            
            # Finally mask z bins
            if k_remaining > 0:
                k_z = min(k_remaining, nbz)
                grad_z[:, :, :k_z] = 0

        # Sum across bins -> grad per axis -> grad_XYZ
        grad_XYZ = torch.cat([
            grad_x.sum(-1, keepdim=True),  # [B, N, 1]
            grad_y.sum(-1, keepdim=True),
            grad_z.sum(-1, keepdim=True),
        ], dim=-1)  # [B, N, 3]

        # Returns: grad for XYZ, None for other arguments (not learnable)
        return grad_XYZ, None, None, None, None


class RotaryPositionEncoding3D(RotaryPositionEncoding):
    # NOTE: adjust inheritance to match your actual parent class

    def __init__(self, feature_dim, pe_type='Rotary3D', rope_type='normal'):
        """
        Args:
            feature_dim: Dimension of the position encoding features
            pe_type: Type of position encoding (default: 'Rotary3D')
            rope_type: Type of RoPE backward pass. Options:
                - 'adam': Use Adam-style normalized gradients (custom backward)
                - 'normal': Use standard PyTorch autograd (normal backward)
                - 'stopgrad': Use stopgrad (zeros first K bins in backward, K passed at runtime)
        """
        super().__init__(feature_dim, pe_type)
        self.feature_dim = feature_dim
        self.rope_type = rope_type.lower()
        
        if self.rope_type not in ['adam', 'normal', 'stopgrad']:
            raise ValueError(f"rope_type must be 'adam', 'normal', or 'stopgrad', got '{rope_type}'")

        # --- ADDED: per-bin Adam second moment buffer (only needed for adam type) ---
        if self.rope_type == 'adam':
            dx = dy = feature_dim // 3
            if dx % 2 == 1:
                dx -= 1
                dy -= 1
            dz = feature_dim - dx - dy
            total_bins = dx // 2 + dy // 2 + dz // 2
            self.register_buffer('adam_v', torch.zeros(total_bins))
        else:
            self.adam_v = None

    def forward(self, XYZ, allow_grad=False, stopgrad_k=0, delta_M=None):
        '''
        @param XYZ: [B,N,3]
        @param allow_grad: whether to allow gradients to flow through
        @param stopgrad_k: number of bins to zero out in backward (for stopgrad rope_type)
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

        # --- ADDED: when allow_grad and rope_type='adam', route through custom Function ---
        if allow_grad and self.rope_type == 'adam':
            return RoPE3DAdamFunction.apply(XYZ, div_term_x, div_term_y, div_term_z, self.adam_v)
        
        # --- ADDED: when allow_grad and rope_type='stopgrad', route through stopgrad Function ---
        if allow_grad and self.rope_type == 'stopgrad':
            return RoPE3DStopGradFunction.apply(
                XYZ, div_term_x, div_term_y, div_term_z, stopgrad_k
            )

        # --- Standard path (either no grad or normal rope_type) ---
        sinx = torch.sin(x_position * div_term_x)  # [B, N, d//6]
        cosx = torch.cos(x_position * div_term_x)
        siny = torch.sin(y_position * div_term_y)
        cosy = torch.cos(y_position * div_term_y)
        sinz = torch.sin(z_position * div_term_z)
        cosz = torch.cos(z_position * div_term_z)

        # Optional: mix sin/cos with delta_M (from cam_token), before view/stack
        if delta_M is not None:
            feat = torch.stack([cosx, cosy, cosz, sinx, siny, sinz], dim=-1)  # [B, N, d//6, 6]
            if delta_M.ndim == 4 and delta_M.shape[-1] == 6:  # (B, N, 6, 6) — per-token 6×6
                feat = torch.einsum('bnci,bnji->bncj', feat, delta_M)
            elif delta_M.ndim == 3 and delta_M.shape[-1] == 6:  # (B, 6, 6) — broadcast 6×6
                feat = torch.einsum('bnci,bji->bncj', feat, delta_M)
            elif delta_M.ndim == 4:  # (B, N, D, D) — per-token D×D
                bsz2, np2, nb, _ = feat.shape
                feat = torch.einsum('bni,bnji->bnj', feat.reshape(bsz2, np2, -1), delta_M).reshape(bsz2, np2, nb, 6)
            else:  # (B, D, D) — broadcast D×D
                bsz2, np2, nb, _ = feat.shape
                feat = torch.einsum('bni,bji->bnj', feat.reshape(bsz2, np2, -1), delta_M).reshape(bsz2, np2, nb, 6)
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

        # Only detach if gradients not allowed (for normal rope_type, grad flows through)
        if not allow_grad:
            position_code = position_code.detach()

        return position_code

    def _compute_sincos_base(self, XYZ, stopgrad_k=0):
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

    def _finalize_from_base(self, base_feat, delta_M=None):
        """Apply optional delta_M to pre-computed sin/cos base and return position_code [B,N,C,2].

        Args:
            base_feat: [B, N, d//6, 6] — output of _compute_sincos_base
            delta_M: optional (B, 6, 6) matrix to mix sin/cos features

        Returns:
            position_code: [B, N, C, 2]
        """
        bsize, npoint = base_feat.shape[:2]

        if delta_M is not None:
            if delta_M.ndim == 4 and delta_M.shape[-1] == 6:  # (B, N, 6, 6) — per-token 6×6
                feat = torch.einsum('bnci,bnji->bncj', base_feat, delta_M)
            elif delta_M.ndim == 3 and delta_M.shape[-1] == 6:  # (B, 6, 6) — broadcast 6×6
                feat = torch.einsum('bnci,bji->bncj', base_feat, delta_M)
            elif delta_M.ndim == 4:  # (B, N, D, D) — per-token D×D
                bsz2, np2, nb, _ = base_feat.shape
                feat = torch.einsum('bni,bnji->bnj', base_feat.reshape(bsz2, np2, -1), delta_M).reshape(bsz2, np2, nb, 6)
            else:  # (B, D, D) — broadcast D×D
                bsz2, np2, nb, _ = base_feat.shape
                feat = torch.einsum('bni,bji->bnj', base_feat.reshape(bsz2, np2, -1), delta_M).reshape(bsz2, np2, nb, 6)
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