import torch
from torch import nn

from ..utils.layers import AttentionModule
from ..utils.position_encodings import RotaryPositionEncoding3D


class RecursiveSetTransformerEncoder(nn.Module):
    """
    Recursive 3D-RoPE Set Transformer Encoder.

    Given N snapshots × ncam cameras of CLIP patch features and unprojected 3D positions,
    runs L recursive blocks (phases A–E) and returns:
      - rgb3d_feats_refined: (B, ncam*P, D)  — mean-pooled across snapshots
      - delta_M:             (B, ncam, 6, 6) — per-camera orthogonal RoPE-mixing matrix

    Block phases:
      A. Compute 3D RoPE from current coordinates (patches only; registers get zero pos).
      B. Cross-view self-attention within each snapshot over [REG_1..ncam, Patches_1..ncam].
      C. Mean-pool register tokens across N snapshots → time-invariant per-camera signal.
      D. Predict per-camera delta_M (6×6 orthogonal, skew-sym → matrix_exp) from pooled regs.
      E. Cross-snapshot set attention: group patches by (cam, patch_id), attend over N
         (no positional encoding → permutation-equivariant w.r.t. snapshot order).

    Phase 1 usage (N=1): pass (B, ncam*P, D); unsqueezed internally. Phases C and E are
    trivially no-ops but the module still provides cross-view attention and delta_M prediction.
    """

    def __init__(
        self,
        embedding_dim: int,
        ncam: int,
        num_layers: int = 2,
        num_attn_heads: int = 8,
        rope_type: str = 'normal',
    ):
        super().__init__()
        self.ncam = ncam
        self.num_layers = num_layers

        # Per-camera learnable register tokens (ncam, D)
        self.register_tokens = nn.Parameter(torch.empty(ncam, embedding_dim))
        nn.init.normal_(self.register_tokens, std=0.02)

        # Shared 3D RoPE (Phase A)
        self.rope = RotaryPositionEncoding3D(embedding_dim, rope_type=rope_type)

        # Phase B: within-snapshot cross-view self-attention (with 3D RoPE)
        self.cross_view_attn = nn.ModuleList([
            AttentionModule(
                num_layers=1, d_model=embedding_dim, n_heads=num_attn_heads,
                is_self=True, rotary_pe=True,
            )
            for _ in range(num_layers)
        ])

        # Phase D: per-camera delta_M predictor (same pattern as _predict_from_cam_feat)
        self.camera_proj = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) for _ in range(num_layers)
        ])
        self.camera_predictor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 36),  # 6×6
            )
            for _ in range(num_layers)
        ])

        # Phase E: cross-snapshot set attention (no positional encoding).
        # seq_len=N (num_history, typically 3) is too small for flash/mem-efficient
        # attention kernels, so force the math backend here only.
        self.cross_snapshot_attn = nn.ModuleList([
            AttentionModule(
                num_layers=1, d_model=embedding_dim, n_heads=num_attn_heads,
                is_self=True, rotary_pe=False, force_math=True,
            )
            for _ in range(num_layers)
        ])

    def _predict_delta_M(self, master_regs: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """(B, ncam, D) → (B, ncam, 6, 6) orthogonal matrix."""
        h = self.camera_proj[layer_idx](master_regs)
        A_skew = self.camera_predictor[layer_idx](h).reshape(*master_regs.shape[:-1], 6, 6)
        A = A_skew - A_skew.transpose(-1, -2)
        # Clamp Frobenius norm to stabilise matrix_exp (mirrors base_denoise_actor)
        max_norm = 3.0
        norm = torch.linalg.norm(A, ord='fro', dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        A = A * (norm.clamp(max=max_norm) / norm)
        return torch.linalg.matrix_exp(A)

    def forward(
        self,
        rgb3d_feats: torch.Tensor,
        rgb3d_pos: torch.Tensor,
        ncam: int | None = None,
    ):
        """
        Args:
            rgb3d_feats: (B, ncam*P, D) or (B, N, ncam*P, D)
            rgb3d_pos:   (B, ncam*P, 3) or (B, N, ncam*P, 3)
            ncam:        override self.ncam if needed at runtime
        Returns:
            rgb3d_feats_refined: (B, ncam*P, D)
            delta_M:             (B, ncam, 6, 6)
        """
        if ncam is None:
            ncam = self.ncam

        # Normalise to (B, N, ncam*P, D) — unsqueeze for single-snapshot (Phase 1)
        if rgb3d_feats.ndim == 3:
            rgb3d_feats = rgb3d_feats.unsqueeze(1)
            rgb3d_pos = rgb3d_pos.unsqueeze(1)

        B, N, NP, D = rgb3d_feats.shape
        if not hasattr(self, '_printed_N'):
            print(f"[RecursiveSetEncoder] B={B} N={N} NP={NP} D={D}")
            self._printed_N = True
        delta_M = None
        patches = rgb3d_feats  # (B, N, NP, D)

        for l in range(self.num_layers):
            # ── Phase A: 3D RoPE ─────────────────────────────────────────────
            # Expand per-camera delta_M (B, ncam, 6, 6) to per-token (B*N, NP, 6, 6)
            delta_M_expanded = None
            if delta_M is not None:
                P = NP // ncam
                dense_cam_ids = torch.arange(ncam, device=delta_M.device).repeat_interleave(P)
                delta_M_per_tok = delta_M[:, dense_cam_ids, :, :]            # (B, NP, 6, 6)
                delta_M_expanded = (
                    delta_M_per_tok.unsqueeze(1)
                    .expand(-1, N, -1, -1, -1)
                    .reshape(B * N, NP, 6, 6)
                )

            rgb3d_pos_flat = rgb3d_pos.reshape(B * N, NP, 3)
            patch_pos = self.rope(rgb3d_pos_flat, delta_M=delta_M_expanded)  # (B*N, NP, D, 2)

            # Registers get zero positional encoding (no 3D location)
            reg_pos = torch.zeros(
                B * N, ncam, patch_pos.shape[-2], patch_pos.shape[-1],
                device=patch_pos.device, dtype=patch_pos.dtype,
            )
            seq_pos = torch.cat([reg_pos, patch_pos], dim=1)                 # (B*N, ncam+NP, D, 2)

            # ── Phase B: cross-view self-attention ────────────────────────────
            regs = self.register_tokens.unsqueeze(0).expand(B * N, -1, -1)   # (B*N, ncam, D)
            patches_flat = patches.reshape(B * N, NP, D)
            seq = torch.cat([regs, patches_flat], dim=1)                      # (B*N, ncam+NP, D)

            seq_out = self.cross_view_attn[l](
                seq, seq, seq1_pos=seq_pos, seq2_pos=seq_pos
            )[-1]  # (B*N, ncam+NP, D)

            regs_out = seq_out[:, :ncam, :].reshape(B, N, ncam, D)           # (B, N, ncam, D)
            patches = seq_out[:, ncam:, :].reshape(B, N, NP, D)              # (B, N, NP, D)

            # ── Phase C: register time-pooling ───────────────────────────────
            master_regs = regs_out.mean(dim=1)                                # (B, ncam, D)

            # ── Phase D: predict delta_M ──────────────────────────────────────
            delta_M = self._predict_delta_M(master_regs, l)                   # (B, ncam, 6, 6)

            # ── Phase E: cross-snapshot set attention ─────────────────────────
            # Group by spatial location (cam_id × patch_id) across N snapshots
            # (B, N, NP, D) → (B*NP, N, D) then attend over N.
            # Skip when N==1 (single snapshot): self-attention over one token is identity.
            if N > 1:
                patches_grouped = patches.permute(0, 2, 1, 3).reshape(B * NP, N, D)
                patches_grouped = self.cross_snapshot_attn[l](
                    patches_grouped, patches_grouped
                )[-1]  # (B*NP, N, D)
                patches = patches_grouped.reshape(B, NP, N, D).permute(0, 2, 1, 3)  # (B, N, NP, D)

        # Aggregate across snapshots (mean pool)
        rgb3d_feats_refined = patches.mean(dim=1)  # (B, NP, D)
        return rgb3d_feats_refined, delta_M
