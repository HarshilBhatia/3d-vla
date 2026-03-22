"""3D Rotary Position Embeddings for cross-attention over spatially grounded image tokens.

Frequencies are computed over the full embedding dim D (interleaved x/y/z triplets).
Different attention heads see different frequency slices, giving multi-scale 3D encoding.

Interleaved layout across D dims:
  dims [0,1]   → x at freq_0,   dims [2,3] → y at freq_0,   dims [4,5] → z at freq_0
  dims [6,7]   → x at freq_1,   dims [8,9] → y at freq_1,   dims [10,11] → z at freq_1
  ...
  num_triplets = D // 6 unique frequencies per axis (256 for D=1536 vs 8 previously).

Text and wrist tokens get position [0,0,0] → cos=1, sin=0 → identity rotation.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def compute_rope_cos_sin(
    positions_3d: torch.Tensor,           # [B, seq_len, 3]
    precomputed_freqs: torch.Tensor,       # [num_triplets] — precomputed, on correct device
    total_dim: int,                        # full embedding dim (1536)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute interleaved 3D RoPE cos/sin for each token position.

    Frequencies are interleaved across the full embedding dim so different heads
    see different frequency scales (head 0 = low-freq/global, head H-1 = high-freq).

    Args:
        positions_3d: [B, seq_len, 3] world-frame coordinates (meters).
        precomputed_freqs: [num_triplets] frequency bands, already on the correct device.
            Computed as: 1 / (base ** (i / num_triplets)) for i in [0, num_triplets).
        total_dim: full embedding dimension, must be divisible by 6.

    Returns:
        cos: [B, seq_len, total_dim // 2]   layout: [cos_x0, cos_y0, cos_z0, cos_x1, ...]
        sin: [B, seq_len, total_dim // 2]
    """
    if total_dim % 6 != 0:
        raise ValueError(f"total_dim={total_dim} must be divisible by 6 for interleaved 3D RoPE")
    num_triplets = total_dim // 6  # 256 for total_dim=1536
    if precomputed_freqs.shape[0] != num_triplets:
        raise ValueError(
            f"precomputed_freqs length {precomputed_freqs.shape[0]} != num_triplets {num_triplets}"
        )

    B, S, _ = positions_3d.shape
    freqs = precomputed_freqs.to(dtype=positions_3d.dtype)

    # Angles per axis: [B, S, num_triplets]
    angles_x = positions_3d[..., 0:1] * freqs   # broadcast: [B, S, 1] * [num_triplets]
    angles_y = positions_3d[..., 1:2] * freqs
    angles_z = positions_3d[..., 2:3] * freqs

    cos_x, cos_y, cos_z = angles_x.cos(), angles_y.cos(), angles_z.cos()
    sin_x, sin_y, sin_z = angles_x.sin(), angles_y.sin(), angles_z.sin()

    # Interleave triplets: stack → [B, S, num_triplets, 3] → reshape → [B, S, total_dim // 2]
    cos = torch.stack([cos_x, cos_y, cos_z], dim=-1).reshape(B, S, total_dim // 2)
    sin = torch.stack([sin_x, sin_y, sin_z], dim=-1).reshape(B, S, total_dim // 2)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dim by splitting in half: [-x2, x1]."""
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def apply_3d_rope_to_keys(
    K: torch.Tensor,   # [B, num_heads, seq_len, head_dim]
    cos: torch.Tensor, # [B, seq_len, total_dim // 2]  where total_dim = num_heads * head_dim
    sin: torch.Tensor, # [B, seq_len, total_dim // 2]
) -> torch.Tensor:
    """Apply interleaved 3D RoPE rotation to key vectors.

    cos/sin are computed over the full embedding dim (interleaved x/y/z triplets).
    Each head h gets the frequency slice [h*head_dim//2 : (h+1)*head_dim//2], which
    corresponds to 8 consecutive (x,y,z) triplets at distinct frequency scales.

    Args:
        K:   [B, num_heads, seq_len, head_dim]
        cos: [B, seq_len, total_dim // 2]   total_dim = num_heads * head_dim
        sin: [B, seq_len, total_dim // 2]

    Returns:
        K_rot: same shape as K
    """
    B, H, S, head_dim = K.shape
    total_dim = H * head_dim
    if cos.shape[-1] != total_dim // 2:
        raise ValueError(
            f"cos last dim {cos.shape[-1]} != total_dim//2 {total_dim // 2} "
            f"(H={H}, head_dim={head_dim})"
        )

    # Reshape: [B, S, total_dim//2] → [B, S, H, head_dim//2] → [B, H, S, head_dim//2]
    cos_h = cos.view(B, S, H, head_dim // 2).permute(0, 2, 1, 3)  # [B, H, S, head_dim//2]
    sin_h = sin.view(B, S, H, head_dim // 2).permute(0, 2, 1, 3)

    # Expand from head_dim//2 to head_dim via repeat (GPT-NeoX rotate_half convention)
    cos_full = cos_h.repeat(1, 1, 1, 2)  # [B, H, S, head_dim]
    sin_full = sin_h.repeat(1, 1, 1, 2)

    return K * cos_full - _rotate_half(K) * sin_full


def apply_delta_m_to_rope(
    rope_cos: torch.Tensor,                    # [B, S, D//2]
    rope_sin: torch.Tensor,                    # [B, S, D//2]
    delta_ms: List[torch.Tensor],              # list of [B, 6, 6], one per camera
    cam_token_indices: List[torch.Tensor],     # list of LongTensors, per-camera token positions in seq
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply per-camera orthogonal 6×6 deltaM rotation to the RoPE cos/sin features.

    The interleaved layout of rope_cos/sin is [cos_x0, cos_y0, cos_z0, cos_x1, ...].
    Reshaping to [B, S, num_triplets, 3] gives [cos_x, cos_y, cos_z] per frequency bin.
    Concatenating cos and sin gives a 6-component feature [cos_x, cos_y, cos_z, sin_x, sin_y, sin_z]
    per frequency bin, which deltaM rotates.

    Args:
        rope_cos: [B, S, D//2] — RoPE cosines (interleaved x/y/z triplets).
        rope_sin: [B, S, D//2] — RoPE sines.
        delta_ms: list of num_cameras × [B, 6, 6] orthogonal matrices.
        cam_token_indices: list of num_cameras LongTensors, each with token positions in [0, S).

    Returns:
        Modified (rope_cos, rope_sin) with per-camera deltaM applied.
    """
    if len(delta_ms) != len(cam_token_indices):
        raise ValueError(
            f"len(delta_ms)={len(delta_ms)} != len(cam_token_indices)={len(cam_token_indices)}"
        )

    B, S, half_D = rope_cos.shape
    if half_D % 3 != 0:
        raise ValueError(f"rope_cos last dim {half_D} must be divisible by 3 (interleaved x/y/z triplets)")
    num_triplets = half_D // 3  # = D // 6, e.g. 256 for D=1536

    rope_cos = rope_cos.clone()
    rope_sin = rope_sin.clone()

    for delta_m, token_indices in zip(delta_ms, cam_token_indices):
        if len(token_indices) == 0:
            continue
        n_c = len(token_indices)

        # Extract per-camera cos/sin and reshape to [B, n_c, num_triplets, 3]
        cos_c = rope_cos[:, token_indices, :].reshape(B, n_c, num_triplets, 3)
        sin_c = rope_sin[:, token_indices, :].reshape(B, n_c, num_triplets, 3)

        # Stack to [B, n_c, num_triplets, 6]: [cos_x, cos_y, cos_z, sin_x, sin_y, sin_z]
        feat = torch.cat([cos_c, sin_c], dim=-1)  # [B, n_c, num_triplets, 6]

        # Apply per-camera 6×6 deltaM: rotate each freq bin's 6-component feature
        # einsum 'bnci,bji->bncj': for each batch and freq bin, apply [B, 6, 6] matrix
        feat = torch.einsum("bnci,bji->bncj", feat, delta_m.to(feat.dtype))

        # Write back — cast to rope_cos dtype to handle autocast contexts
        rope_cos[:, token_indices, :] = feat[..., :3].reshape(B, n_c, half_D).to(rope_cos.dtype)
        rope_sin[:, token_indices, :] = feat[..., 3:].reshape(B, n_c, half_D).to(rope_sin.dtype)

    return rope_cos, rope_sin


def apply_action_delta_m_to_rope(
    rope_cos_q: torch.Tensor,   # [B, seq_q, D//2]
    rope_sin_q: torch.Tensor,   # [B, seq_q, D//2]
    delta_m: torch.Tensor,      # [B, 6, 6]
    action_start: int,          # first action token index (1, after state token)
    action_end: int,            # one past last action token index (T, before register tokens)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a single orthogonal 6×6 deltaM to action token query RoPE values.

    Operates on the contiguous slice [action_start:action_end] of the query sequence,
    leaving the state token (index 0) and register tokens (indices T:) unchanged.

    Args:
        rope_cos_q: [B, seq_q, D//2] — query RoPE cosines.
        rope_sin_q: [B, seq_q, D//2] — query RoPE sines.
        delta_m:    [B, 6, 6] orthogonal matrix from ActionDeltaMPredictor.
        action_start: first action token index (typically 1).
        action_end:   one past last action token index (typically T = 1 + action_horizon).

    Returns:
        Modified (rope_cos_q, rope_sin_q) with deltaM applied to action token slice.
    """
    B, seq_q, half_D = rope_cos_q.shape
    if half_D % 3 != 0:
        raise ValueError(
            f"rope_cos_q last dim {half_D} must be divisible by 3 (interleaved x/y/z triplets)"
        )
    n_act = action_end - action_start
    if n_act <= 0:
        raise ValueError(f"action_end={action_end} must be > action_start={action_start}")

    num_triplets = half_D // 3

    rope_cos_q = rope_cos_q.clone()
    rope_sin_q = rope_sin_q.clone()

    # Extract action slice and reshape to [B, n_act, num_triplets, 3]
    cos_a = rope_cos_q[:, action_start:action_end, :].reshape(B, n_act, num_triplets, 3)
    sin_a = rope_sin_q[:, action_start:action_end, :].reshape(B, n_act, num_triplets, 3)

    # Stack to [B, n_act, num_triplets, 6]: [cos_x, cos_y, cos_z, sin_x, sin_y, sin_z]
    feat = torch.cat([cos_a, sin_a], dim=-1)

    # Apply [B, 6, 6] deltaM per frequency bin
    feat = torch.einsum("bnci,bji->bncj", feat, delta_m.to(feat.dtype))

    # Write back
    rope_cos_q[:, action_start:action_end, :] = feat[..., :3].reshape(B, n_act, half_D).to(rope_cos_q.dtype)
    rope_sin_q[:, action_start:action_end, :] = feat[..., 3:].reshape(B, n_act, half_D).to(rope_sin_q.dtype)

    return rope_cos_q, rope_sin_q


class RoPE3DCrossAttnProcessor:
    """Custom diffusers attention processor that applies 3D RoPE to keys in cross-attention.

    When rope_cos/rope_sin are None (e.g. text-token blocks, or training without pos_cache_dir),
    this processor is equivalent to standard scaled dot-product attention.

    Set on each cross-attention Attention module via:
        block.attn1.set_processor(RoPE3DCrossAttnProcessor())
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        rope_cos_q: Optional[torch.Tensor] = None,
        rope_sin_q: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        B, seq_len_q, _ = hidden_states.shape

        # Project Q, K, V
        query = attn.to_q(hidden_states)
        kv_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(kv_states)
        value = attn.to_v(kv_states)

        # Reshape to multi-head format: [B, heads, S, head_dim]
        head_dim = key.shape[-1] // attn.heads
        query = query.view(B, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key.view(B, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(B, -1, attn.heads, head_dim).transpose(1, 2)

        # Rotate K with R(p_k): encodes absolute image token positions
        if rope_cos is not None and rope_sin is not None:
            key = apply_3d_rope_to_keys(key, rope_cos.to(key.dtype), rope_sin.to(key.dtype))

        # Rotate Q[state token] with R(p_eef): gives relative attention Q·R(p_k - p_eef)·K
        if rope_cos_q is not None and rope_sin_q is not None:
            query = apply_3d_rope_to_keys(query, rope_cos_q.to(query.dtype), rope_sin_q.to(query.dtype))

        # Convert bool attention mask [B, S_k] to broadcastable format for SDPA
        attn_bias: Optional[torch.Tensor] = None
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                # [B, S_k] → [B, 1, 1, S_k]; SDPA interprets True=attend
                attn_bias = attention_mask.unsqueeze(1).unsqueeze(1)
            else:
                attn_bias = attention_mask

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_bias, dropout_p=0.0, is_causal=False
        )

        # Reshape back and apply output projection
        hidden_states = hidden_states.transpose(1, 2).reshape(B, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
