"""
Cross-modal alignment module: learnable orthogonal matrix ΔM via skew-symmetric
exponential parameterization. Used to align vision and trajectory representations
in cross-modal attention blocks.
"""
import torch
from torch import nn


class CrossModalAlignment(nn.Module):
    """
    Learnable orthogonal cross-modal alignment matrix ΔM ∈ SO(d).

    Parameterization: B ∈ ℝ^{d×d} learnable, A = B - B^T (skew-symmetric),
    ΔM = matrix_exp(A). With B initialized to zeros, ΔM = I initially.

    Supports optional per-head mode: B shape (num_heads, head_dim, head_dim).
    Default: shared ΔM across heads with B shape (head_dim, head_dim).
    """

    def __init__(self, head_dim, num_heads=None, per_head=False):
        """
        Args:
            head_dim: dimension per head (d)
            num_heads: total number of heads (required if per_head=True)
            per_head: if True, use per-head ΔM; else shared across heads
        """
        super().__init__()
        self.head_dim = head_dim
        self.per_head = per_head
        if per_head and num_heads is None:
            raise ValueError("num_heads required when per_head=True")
        self.num_heads = num_heads

        if per_head:
            # B: (num_heads, head_dim, head_dim)
            self.B = nn.Parameter(torch.zeros(num_heads, head_dim, head_dim))
        else:
            # B: (head_dim, head_dim)
            self.B = nn.Parameter(torch.zeros(head_dim, head_dim))

    def forward(self, lambda_reg=0.0):
        """
        Compute ΔM = matrix_exp(A) where A = B - B^T.

        Returns:
            delta_M: (head_dim, head_dim) or (num_heads, head_dim, head_dim)
            reg: optional regularization ||A||_F^2 if lambda_reg > 0
        """
        if self.per_head:
            # A = B - B^T, shape (H, d, d)
            A = self.B - self.B.transpose(-2, -1)
            delta_M = torch.linalg.matrix_exp(A)
            reg = (A * A).sum() if lambda_reg > 0 else torch.tensor(0.0, device=A.device, dtype=A.dtype)
        else:
            A = self.B - self.B.T
            delta_M = torch.linalg.matrix_exp(A)
            reg = (A * A).sum() if lambda_reg > 0 else torch.tensor(0.0, device=A.device, dtype=A.dtype)
        return delta_M, reg

    def get_logging_stats(self, lambda_reg=0.0):
        """
        Compute logging statistics for training monitoring.

        Returns:
            dict with: frob_A, spectral_delta_M, det_delta_M, reg
        """
        with torch.no_grad():
            if self.per_head:
                A = self.B - self.B.transpose(-2, -1)
            else:
                A = self.B - self.B.T
            delta_M, reg = self.forward(lambda_reg=lambda_reg)

            frob_A = (A * A).sum().sqrt().item()
            # Spectral norm = largest singular value
            if delta_M.dim() == 2:
                spectral = torch.linalg.norm(delta_M, ord=2).item()
                det_val = torch.linalg.det(delta_M).item()
            else:
                spectral = torch.linalg.norm(delta_M, ord=2, dim=(-2, -1)).max().item()
                det_val = torch.linalg.det(delta_M).mean().item()
            return {
                "cross_modal_frob_A": frob_A,
                "cross_modal_spectral_delta_M": spectral,
                "cross_modal_det_delta_M": det_val,
                "cross_modal_reg": reg.item() if isinstance(reg, torch.Tensor) and reg.numel() == 1 else reg,
            }
