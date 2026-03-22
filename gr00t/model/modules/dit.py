from typing import Optional

from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import SinusoidalPositionalEmbedding, TimestepEmbedding, Timesteps
import torch
from torch import nn
import torch.nn.functional as F


class DeltaMPredictor(nn.Module):
    """Per-image learnable orthogonal 6×6 RoPE rotation (deltaM).

    Converts a per-camera register token [B, D] into an orthogonal 6×6 matrix
    via: LayerNorm → SwiGLU → Linear(D→36) → skew-sym → Frobenius-norm clip → matrix_exp.

    Architecture matches ActionDeltaMPredictor:
    - LayerNorm: decouples head from varying register token scale.
    - SwiGLU (no bias): multiplicative gate selectively activates Lie algebra axes.
    - No expansion (hidden=d_token): output is only 36 dims.
    - Zero-init output weight: A=0 at init → deltaM=I (identity residual behavior).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_cameras: int,
        backbone_dim: int = 2048,
        max_norm: float = 3.0,
    ):
        super().__init__()
        if num_cameras < 1:
            raise ValueError(f"num_cameras={num_cameras} must be >= 1")
        self.num_cameras = num_cameras
        self.max_norm = max_norm
        # Project backbone thumbnail to DiT hidden dim (default init).
        # Register tokens start as meaningful projections of backbone features,
        # ensuring gradients flow through thumbnail_proj from step 0.
        self.thumbnail_proj = nn.Linear(backbone_dim, hidden_dim)
        # Per-camera SwiGLU heads: LayerNorm → SwiGLU (no bias) → Linear(D→36, zero-init)
        self.norms   = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_cameras)])
        self.w_gates = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_cameras)])
        self.w_vals  = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_cameras)])
        self.w_outs  = nn.ModuleList([nn.Linear(hidden_dim, 36, bias=False) for _ in range(num_cameras)])
        for w_out in self.w_outs:
            nn.init.zeros_(w_out.weight)

    def init_register_tokens(self, thumbnails: torch.Tensor) -> torch.Tensor:
        """Project thumbnails to DiT space.

        Args:
            thumbnails: [B, num_cameras, backbone_dim]

        Returns:
            register_tokens: [B, num_cameras, hidden_dim]
        """
        if thumbnails.shape[1] != self.num_cameras:
            raise ValueError(
                f"thumbnails.shape[1]={thumbnails.shape[1]} != num_cameras={self.num_cameras}"
            )
        return self.thumbnail_proj(thumbnails)

    def compute_delta_m(self, register_tokens: torch.Tensor) -> list:
        """Compute per-camera orthogonal 6×6 deltaM matrices.

        Args:
            register_tokens: [B, num_cameras, hidden_dim]

        Returns:
            list of num_cameras × Tensor[B, 6, 6] orthogonal matrices
        """
        if register_tokens.shape[1] != self.num_cameras:
            raise ValueError(
                f"register_tokens.shape[1]={register_tokens.shape[1]} != num_cameras={self.num_cameras}"
            )
        delta_ms = []
        for c in range(self.num_cameras):
            x = self.norms[c](register_tokens[:, c, :])
            x = F.silu(self.w_gates[c](x)) * self.w_vals[c](x)  # SwiGLU
            A = self.w_outs[c](x).reshape(-1, 6, 6)              # [B, 6, 6]
            A = A - A.transpose(-1, -2)                           # skew-symmetric
            frob = A.norm(dim=(-2, -1), keepdim=True)
            safe_frob = frob.clamp(min=1e-8)
            A = A * (frob.clamp(max=self.max_norm) / safe_frob)
            delta_ms.append(torch.linalg.matrix_exp(A))          # [B, 6, 6]
        return delta_ms


class ActionDeltaMPredictor(nn.Module):
    """State-token-conditioned orthogonal 6×6 RoPE rotation for action token queries.

    Architecture:
      LayerNorm → SwiGLU (hidden=d_token, no bias) → Linear(d_token→36, no bias, zero-init)
      → skew-sym → Frobenius-norm clip → matrix_exp → orthogonal [B, 6, 6]

    Design choices:
    - LayerNorm: decouples head training from varying token scale across training.
    - SwiGLU: multiplicative gate selectively suppresses near-zero rotation axes.
    - No expansion (hidden = d_token): output is only 36 dims; expansion would overfit.
    - No bias in gate/val: LayerNorm handles centering.
    - Zero-init output: A_flat = 0 at init → deltaM = I (identity residual behavior).
    """

    def __init__(self, hidden_dim: int, max_norm: float = 3.0):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim={hidden_dim} must be >= 1")
        self.max_norm = max_norm
        self.norm = nn.LayerNorm(hidden_dim)
        # SwiGLU gate and value projections — no bias (LayerNorm handles centering)
        self.w_gate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_val  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Output: d_token → 36 scalars for 6×6 Lie algebra element — no bias, zero-init
        self.w_out  = nn.Linear(hidden_dim, 36, bias=False)
        nn.init.zeros_(self.w_out.weight)

    def forward(self, state_token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_token: [B, hidden_dim] — current state token embedding.

        Returns:
            delta_m: [B, 6, 6] orthogonal matrix.
        """
        x = self.norm(state_token)
        x = F.silu(self.w_gate(x)) * self.w_val(x)    # SwiGLU
        B = x.shape[0]
        A = self.w_out(x).reshape(B, 6, 6)             # [B, 6, 6]
        A = A - A.transpose(-1, -2)                    # skew-symmetric
        frob = A.norm(dim=(-2, -1), keepdim=True)
        safe_frob = frob.clamp(min=1e-8)
        A = A * (frob.clamp(max=self.max_norm) / safe_frob)
        return torch.linalg.matrix_exp(A)              # [B, 6, 6] orthogonal


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        rope_base_freq: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.rope_base_freq = rope_base_freq
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        if final_dropout:
            self.final_dropout = nn.Dropout(dropout)
        else:
            self.final_dropout = None

        # Precompute 3D RoPE frequency table over the full embedding dim (interleaved layout).
        # num_triplets = dim // 6 unique frequencies per axis (256 for dim=1536).
        if dim % 6 != 0:
            raise ValueError(f"dim={dim} must be divisible by 6 for interleaved 3D RoPE")
        num_triplets = dim // 6
        rope_freqs = 1.0 / (
            rope_base_freq ** (torch.arange(num_triplets, dtype=torch.float32) / num_triplets)
        )
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        rope_cos_q: Optional[torch.Tensor] = None,
        rope_sin_q: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 0. Self-Attention / Cross-Attention
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # All RoPE cos/sin are pre-computed by AlternateVLDiT and passed in directly.
        cross_attention_kwargs = {}
        if encoder_hidden_states is not None:
            if rope_cos is not None and rope_sin is not None:
                cross_attention_kwargs["rope_cos"] = rope_cos
                cross_attention_kwargs["rope_sin"] = rope_sin
            if rope_cos_q is not None and rope_sin_q is not None:
                cross_attention_kwargs["rope_cos_q"] = rope_cos_q
                cross_attention_kwargs["rope_sin_q"] = rope_sin_q

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=(
                encoder_attention_mask if encoder_hidden_states is not None else attention_mask
            ),
            **cross_attention_kwargs,
        )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class DiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
        cross_attention_dim: Optional[int] = None,
        rope_position_noise_std: float = 0.0,
        rope_base_freq: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        # Timestep encoder
        self.timestep_encoder = TimestepEncoder(
            embedding_dim=self.inner_dim, compute_dtype=self.compute_dtype
        )

        all_blocks = []
        for idx in range(self.config.num_layers):
            use_self_attn = idx % 2 == 1 and interleave_self_attention
            curr_cross_attention_dim = cross_attention_dim if not use_self_attn else None

            all_blocks += [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                    rope_base_freq=rope_base_freq,
                )
            ]
        self.transformer_blocks = nn.ModuleList(all_blocks)

        # Output blocks
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.output_dim)
        print(
            "Total number of DiT parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states: torch.Tensor,  # Shape: (B, S, D)
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
    ):
        # Encode timesteps
        temb = self.timestep_encoder(timestep)

        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        # Output processing
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)


class AlternateVLDiT(DiT):
    """
    Alternate Vision-Language DiT that separates image and non-image tokens
    during cross-attention processing.
    """

    def __init__(
        self,
        *args,
        attend_text_every_n_blocks: int = 2,
        use_state_eef_rope: bool = False,
        use_action_eef_rope: bool = False,
        use_delta_m: bool = False,
        num_cameras: int = 2,
        use_action_delta_m: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.attend_text_every_n_blocks = attend_text_every_n_blocks
        self.use_state_eef_rope = use_state_eef_rope
        self.use_action_eef_rope = use_action_eef_rope
        self.use_delta_m = use_delta_m
        self.num_cameras = num_cameras
        self.use_action_delta_m = use_action_delta_m
        # Install 3D RoPE processor on all cross-attention blocks (backward-compatible:
        # behaves as standard SDPA when rope_cos/rope_sin are not passed).
        from gr00t.model.gr00t_n1d6.rope_3d import RoPE3DCrossAttnProcessor
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 0:  # cross-attention blocks
                block.attn1.set_processor(RoPE3DCrossAttnProcessor())

        if use_delta_m:
            # backbone_dim = cross_attention_dim (stored in self.config after super().__init__)
            backbone_dim = self.config.cross_attention_dim or 2048
            self.delta_m_pred = DeltaMPredictor(
                hidden_dim=self.inner_dim,
                num_cameras=num_cameras,
                backbone_dim=backbone_dim,
            )

        if use_action_delta_m:
            self.action_delta_m_pred = ActionDeltaMPredictor(hidden_dim=self.inner_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states: torch.Tensor,  # Shape: (B, S, backbone_dim)
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
        image_mask: Optional[torch.Tensor] = None,
        backbone_attention_mask: Optional[torch.Tensor] = None,
        token_positions_3d: Optional[torch.Tensor] = None,
        eef_position_3d: Optional[torch.Tensor] = None,
        camera_positions_3d: Optional[torch.Tensor] = None,  # [B, num_cameras, 3]
    ):
        assert image_mask is not None, "Image mask is required"

        if self.use_delta_m and token_positions_3d is None:
            raise ValueError("use_delta_m=True requires token_positions_3d (--use-3d-rope)")
        if self.use_action_delta_m and token_positions_3d is None:
            raise ValueError("use_action_delta_m=True requires token_positions_3d (--use-3d-rope)")

        # Encode timesteps
        temb = self.timestep_encoder(timestep)

        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        # Create attention masks for image and non-image tokens
        # image_mask shape: (B, S) where True indicates image tokens
        # For attention, we need to invert: False means "don't attend to this token"

        image_attention_mask = image_mask & backbone_attention_mask
        non_image_attention_mask = (~image_mask) & backbone_attention_mask

        # Augment 3D positions with Gaussian noise (training and eval when std > 0)
        if token_positions_3d is not None and self.config.rope_position_noise_std > 0.0:
            token_positions_3d = token_positions_3d + torch.randn_like(token_positions_3d) * self.config.rope_position_noise_std

        # --- DeltaM setup: compute register tokens and per-camera token indices ---
        T = hidden_states.shape[1]  # original number of query tokens (state + action)
        K = self.num_cameras        # number of register tokens to append
        cam_token_indices = None    # per-camera token indices in encoder sequence
        current_delta_ms = None     # computed after each image block

        if self.use_delta_m:
            # Find per-camera token positions in encoder sequence (use first batch item;
            # all items in a batch share the same embodiment and token layout).
            img_positions = image_mask[0].nonzero(as_tuple=True)[0]  # [total_image_tokens]
            total_img = len(img_positions)
            if total_img % K != 0:
                raise ValueError(
                    f"Total image tokens {total_img} not divisible by num_cameras={K}"
                )
            tpc = total_img // K  # tokens per camera
            cam_token_indices = [img_positions[c * tpc : (c + 1) * tpc] for c in range(K)]

            # Compute per-camera thumbnails: mean-pool each camera's backbone features
            thumbnails = torch.stack(
                [encoder_hidden_states[:, cam_token_indices[c], :].mean(dim=1) for c in range(K)],
                dim=1,
            )  # [B, K, backbone_dim]

            # Project thumbnails → register tokens in DiT hidden space [B, K, D]
            register_tokens = self.delta_m_pred.init_register_tokens(thumbnails)

            # Append register tokens to action/state hidden states
            extended_hidden = torch.cat([hidden_states, register_tokens], dim=1)  # [B, T+K, D]
        else:
            extended_hidden = hidden_states

        all_hidden_states = [hidden_states]
        assert self.config.interleave_self_attention, "Interleave self attention must be enabled"

        # Precompute base RoPE cos/sin from token_positions_3d (reused per image block)
        base_rope_cos = None
        base_rope_sin = None
        if token_positions_3d is not None:
            from gr00t.model.gr00t_n1d6.rope_3d import compute_rope_cos_sin
            base_rope_cos, base_rope_sin = compute_rope_cos_sin(
                token_positions_3d,
                precomputed_freqs=self.transformer_blocks[0].rope_freqs,
                total_dim=self.inner_dim,
            )

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                # Self-attention blocks — all tokens (including register tokens) attend together
                extended_hidden = block(
                    extended_hidden,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                is_image_block = idx % (2 * self.attend_text_every_n_blocks) != 0

                if not is_image_block:
                    # Text cross-attn: only first T tokens attend (register tokens excluded)
                    hs_out = block(
                        extended_hidden[:, :T, :],
                        attention_mask=None,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=non_image_attention_mask,
                        temb=temb,
                    )
                    # Preserve register tokens unchanged
                    extended_hidden = torch.cat(
                        [hs_out, extended_hidden[:, T:, :]], dim=1
                    )
                else:
                    # Image cross-attn: full extended_hidden (register tokens also attend to image)
                    # Apply deltaM from previous image block to base rope (if available)
                    curr_rope_cos = base_rope_cos
                    curr_rope_sin = base_rope_sin
                    if (
                        self.use_delta_m
                        and current_delta_ms is not None
                        and base_rope_cos is not None
                    ):
                        from gr00t.model.gr00t_n1d6.rope_3d import apply_delta_m_to_rope
                        curr_rope_cos, curr_rope_sin = apply_delta_m_to_rope(
                            base_rope_cos, base_rope_sin, current_delta_ms, cam_token_indices
                        )

                    # Build query position tensor [B, T+K, 3] then compute query RoPE.
                    # All query RoPE is pre-computed here and passed directly to the block.
                    rope_cos_q = None
                    rope_sin_q = None
                    if curr_rope_cos is not None:
                        from gr00t.model.gr00t_n1d6.rope_3d import compute_rope_cos_sin
                        B_val = extended_hidden.shape[0]
                        seq_q = extended_hidden.shape[1]  # T+K (or T if not use_delta_m)
                        query_pos_3d = torch.zeros(
                            B_val, seq_q, 3,
                            device=hidden_states.device, dtype=torch.float32,
                        )
                        if eef_position_3d is not None:
                            p_eef = eef_position_3d.to(torch.float32)
                            if self.use_state_eef_rope:
                                query_pos_3d[:, 0, :] = p_eef
                            if self.use_action_eef_rope:
                                query_pos_3d[:, 1:T, :] = p_eef.unsqueeze(1)
                        if self.use_delta_m and camera_positions_3d is not None:
                            query_pos_3d[:, T : T + K, :] = camera_positions_3d.to(torch.float32)
                        rope_cos_q, rope_sin_q = compute_rope_cos_sin(
                            query_pos_3d,
                            precomputed_freqs=self.transformer_blocks[0].rope_freqs,
                            total_dim=self.inner_dim,
                        )
                        if self.use_action_delta_m:
                            from gr00t.model.gr00t_n1d6.rope_3d import apply_action_delta_m_to_rope
                            state_tok = extended_hidden[:, 0, :]  # [B, D], re-read each block
                            action_delta_m = self.action_delta_m_pred(state_tok)  # [B, 6, 6]
                            rope_cos_q, rope_sin_q = apply_action_delta_m_to_rope(
                                rope_cos_q, rope_sin_q, action_delta_m,
                                action_start=1, action_end=T,
                            )

                    extended_hidden = block(
                        extended_hidden,
                        attention_mask=None,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=image_attention_mask,
                        temb=temb,
                        rope_cos=curr_rope_cos,
                        rope_sin=curr_rope_sin,
                        rope_cos_q=rope_cos_q,
                        rope_sin_q=rope_sin_q,
                    )

                    # After image block: compute new deltaM from updated register tokens
                    if self.use_delta_m:
                        current_delta_ms = self.delta_m_pred.compute_delta_m(
                            extended_hidden[:, T:, :]
                        )

            # Append only the first T tokens to all_hidden_states tracking
            all_hidden_states.append(extended_hidden[:, :T, :])

        # Discard register tokens — only keep the original T query tokens
        hidden_states = extended_hidden[:, :T, :]

        # Output processing
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)


class SelfAttentionTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        print(
            "Total number of SelfAttentionTransformer parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        return_all_hidden_states: bool = False,
    ):
        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            hidden_states = block(hidden_states)
            all_hidden_states.append(hidden_states)

        if return_all_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states
