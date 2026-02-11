"""
Learnable RoPE (ComRoPE) for self-attention: block-diagonal rotation from
skew-symmetric exponent matrices and multi-axis positions.
"""
from typing import Optional, Tuple

import torch
from torch import nn


def _default_com_rope_config(
    embed_dim: int,
    num_heads: int,
    block_size: int = 4,
    init_std: float = 0.02,
    num_axes: int = 3,
    dropout: float = 0.0,
):
    """Build a minimal config for ComRoPE (no CLIP dependency)."""
    assert embed_dim % num_heads == 0
    head_dim = embed_dim // num_heads
    assert head_dim % block_size == 0, f"head_dim {head_dim} must be divisible by block_size {block_size}"
    return type("ComRoPEConfig", (), {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "block_size": block_size,
        "init_std": init_std,
        "num_axes": num_axes,
        "dropout": dropout,
    })()


class RoPESelfAttentionBase(nn.Module):
    """
    Self-attention with RoPE applied via block-diagonal rotation matrices R = exp(sum_axes A * x).
    Positions are set per forward via set_positions(positions) with shape (bs, len, num_axes).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        block_size: int = 4,
        num_axes: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size
        self.num_axes = num_axes
        self.scale = self.head_dim ** -0.5

        print(self.head_dim, embed_dim)

        assert self.head_dim % block_size == 0
        self.register_buffer("block_diag_selector", self._init_block_diag_selector(), persistent=False)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.positions: Optional[torch.Tensor] = None

    def _init_block_diag_selector(self) -> torch.Tensor:
        block = torch.ones(self.block_size, self.block_size, dtype=torch.bool)
        eye = torch.eye(self.head_dim // self.block_size, dtype=torch.bool)
        return torch.kron(eye, block)

    def get_exponent_matrix(self) -> torch.Tensor:
        """Return skew-symmetric A with shape (num_heads or 1, num_axes, num_blocks, block_size, block_size)."""
        raise NotImplementedError

    def get_rotation_matrix(self) -> torch.Tensor:
        """R = exp(sum_axes A * positions). Shape: (bs, len, num_heads, head_dim, head_dim)."""
        if self.positions is None:
            raise ValueError("positions must be set before calling get_rotation_matrix")
        A = self.get_exponent_matrix()
        eye = torch.eye(self.block_size, device=A.device, dtype=A.dtype)
        # (bs, len, num_axes) -> (bs, len, num_axes, block_size, block_size)
        positions = self.positions.unsqueeze(-1).unsqueeze(-1) * eye
        # (bs, len, 1, num_axes, 1, block_size, block_size) for matmul with A
        positions = positions.unsqueeze(2).unsqueeze(4)
        A_expand = A.unsqueeze(0).unsqueeze(0)
        lnR = torch.matmul(A_expand, positions)
        lnR = lnR.sum(dim=3)
        R = torch.linalg.matrix_exp(lnR)
        diagR = torch.zeros(*(R.shape[:3]), self.head_dim, self.head_dim, device=A.device, dtype=A.dtype)
        diagR[:, :, :, self.block_diag_selector] = R.reshape(*(R.shape[:3]), -1)
        return diagR

    def set_positions(self, positions: torch.Tensor):
        assert positions.dim() == 3, f"positions must be 3D (batch, len, num_axes), got {positions.shape}"
        assert positions.size(2) == self.num_axes
        self.positions = positions

    def unset_positions(self):
        self.positions = None

    def _shape(self, x: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: (B, T, C). Output: (B, T, C), optional weights."""
        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), tgt_len, bsz)
        value_states = self._shape(self.v_proj(hidden_states), tgt_len, bsz)
        query_states = self._shape(query_states, tgt_len, bsz)

        R = self.get_rotation_matrix()
        R = R.transpose(1, 2)

        Q = query_states.unsqueeze(-1)
        Q = torch.matmul(R, Q).squeeze(-1)
        K = key_states.unsqueeze(-1)
        K = torch.matmul(R, K).squeeze(-1)

        attn_weights = torch.matmul(Q, K.transpose(2, 3))
        src_len = attn_weights.size(2)

        if causal_attention_mask is not None:
            attn_weights = attn_weights + causal_attention_mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights_reshaped = attn_weights if output_attentions else None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout.p, training=self.training)

        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, src_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped

    def reset_pe(self):
        pass


class ComRoPELDAttention(RoPESelfAttentionBase):
    """Learnable RoPE: skew-symmetric exponent matrix from learnable freqs and multiplier."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        block_size: int = 4,
        num_axes: int = 3,
        init_std: float = 0.02,
        dropout: float = 0.0,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            block_size=block_size,
            num_axes=num_axes,
            dropout=dropout,
        )
        self.init_std = init_std
        self.num_blocks = self.head_dim // self.block_size

        self.freqs = nn.Parameter(
            torch.randn(
                self.num_heads,
                1,
                self.num_blocks,
                self.block_size,
                self.block_size,
            ) * self.init_std
        )
        self.multiplier = nn.Parameter(torch.randn(1, self.num_axes, self.num_blocks, 1, 1))

    def get_exponent_matrix(self) -> torch.Tensor:
        skew_symmetric = self.freqs - self.freqs.transpose(-1, -2)
        skew_symmetric = skew_symmetric * self.multiplier
        return skew_symmetric

    def reset_pe(self):
        self.freqs.data.normal_(0, self.init_std)
        self.multiplier.data.normal_(0, 1)
