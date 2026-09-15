"""Simple sampled external--wrist cross-view RoPE diagnostic.

This is isolated from the model and training code.  It implements the actual
repository Delta-M semantics: Delta-M mixes the six sin/cos basis channels
before the final RoPE rotation blocks are formed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional
import math
import torch


@dataclass(frozen=True)
class CrossViewRoPEMetrics:
    pairs: int
    clean: float
    uncorrected: float
    corrected: float

    @property
    def improvement(self) -> float:
        return self.uncorrected - self.corrected

    @property
    def recovered_fraction(self) -> float:
        return self.improvement / self.uncorrected if self.uncorrected else float("nan")


def rope_sincos(xyz: torch.Tensor, feature_dim: int, delta_m: Optional[torch.Tensor] = None):
    """Build the production RoPE cos/sin coefficients for ``xyz: (B,N,3)``.

    ``delta_m``, if supplied, is ``(B,N,6,6)`` and is applied exactly where
    ``RotaryPositionEncoding3D`` applies its per-camera Delta-M.
    """
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError("xyz must have shape (B,N,3)")
    dx = dy = feature_dim // 3
    if dx % 2:
        dx -= 1
        dy -= 1
    dz = feature_dim - dx - dy
    terms = [torch.exp(torch.arange(0, d, 2, device=xyz.device, dtype=xyz.dtype) * (-math.log(10000.0) / d)) for d in (dx, dy, dz)]
    x, y, z = xyz.unbind(-1)
    raw = torch.stack((torch.cos(x[..., None] * terms[0]), torch.cos(y[..., None] * terms[1]), torch.cos(z[..., None] * terms[2]), torch.sin(x[..., None] * terms[0]), torch.sin(y[..., None] * terms[1]), torch.sin(z[..., None] * terms[2])), -1)
    if delta_m is not None:
        if delta_m.shape != (*xyz.shape[:2], 6, 6):
            raise ValueError("delta_m must have shape (B,N,6,6)")
        raw = torch.einsum("bnij,bnkj->bnki", delta_m.to(raw.dtype), raw)
    cos = torch.cat([raw[..., i].repeat_interleave(2, -1) for i in range(3)], -1)
    sin = torch.cat([raw[..., i].repeat_interleave(2, -1) for i in range(3, 6)], -1)
    return cos, sin


def _relative(qc, qs, kc, ks):
    """Coefficient form of the block-diagonal relative operator R(q)^T R(k)."""
    return qc * kc + qs * ks, qc * ks - qs * kc


def _fro_error(qc, qs, kc, ks, target):
    c, s = _relative(qc, qs, kc, ks)
    # Each cos/sin pair is a 2x2 rotation block.
    return (2 * ((c - target[0]).square() + (s - target[1]).square()).sum(-1)).sqrt()


@torch.no_grad()
def cross_view_rope_error(clean_xyz: torch.Tensor, corrupted_xyz: torch.Tensor,
                          camera_ids: torch.Tensor, delta_m: torch.Tensor,
                          feature_dim: int, *, external_ids: Iterable[int] = (0, 1),
                          wrist_ids: Iterable[int] = (2, 3), pairs_per_batch: int = 1024,
                          generator: Optional[torch.Generator] = None) -> CrossViewRoPEMetrics:
    """Average the document's metric over sampled ordinary token pairs.

    Positions and IDs are aligned token-for-token between clean/corrupted
    inputs.  Pairs are sampled external-query × wrist-key; no correspondence,
    nearest-neighbour, visibility, feature, or policy-success computation is
    involved. ``delta_m`` is one captured layer prediction: ``(B,ncam,6,6)``.
    """
    if clean_xyz.shape != corrupted_xyz.shape or clean_xyz.ndim != 3:
        raise ValueError("clean_xyz and corrupted_xyz must share shape (B,N,3)")
    if camera_ids.shape != clean_xyz.shape[:2]:
        raise ValueError("camera_ids must have shape (B,N)")
    if delta_m.ndim != 4 or delta_m.shape[0] != clean_xyz.shape[0] or delta_m.shape[-2:] != (6, 6):
        raise ValueError("delta_m must have shape (B,ncam,6,6)")
    if int(camera_ids.min()) < 0 or int(camera_ids.max()) >= delta_m.shape[1]:
        raise ValueError("camera_ids outside delta_m range")
    clean = rope_sincos(clean_xyz, feature_dim)
    noisy = rope_sincos(corrupted_xyz, feature_dim)
    batch = torch.arange(clean_xyz.shape[0], device=clean_xyz.device)[:, None]
    corrected = rope_sincos(corrupted_xyz, feature_dim, delta_m[batch, camera_ids])
    ext_ids = torch.tensor(tuple(external_ids), device=camera_ids.device)
    wrist_ids = torch.tensor(tuple(wrist_ids), device=camera_ids.device)
    total = torch.zeros(3, device=clean_xyz.device, dtype=torch.float64)
    count = 0
    for b in range(clean_xyz.shape[0]):
        ext = torch.where(torch.isin(camera_ids[b], ext_ids))[0]
        wrist = torch.where(torch.isin(camera_ids[b], wrist_ids))[0]
        if not len(ext) or not len(wrist):
            raise ValueError("each item needs external and wrist tokens")
        n = min(pairs_per_batch, len(ext) * len(wrist))
        flat = torch.randperm(len(ext) * len(wrist), device=camera_ids.device, generator=generator)[:n]
        e, w = ext[flat // len(wrist)], wrist[flat % len(wrist)]
        target = _relative(clean[0][b, e], clean[1][b, e], clean[0][b, w], clean[1][b, w])
        total[0] += _fro_error(clean[0][b, e], clean[1][b, e], clean[0][b, w], clean[1][b, w], target).sum()
        total[1] += _fro_error(noisy[0][b, e], noisy[1][b, e], clean[0][b, w], clean[1][b, w], target).sum()
        total[2] += _fro_error(corrected[0][b, e], corrected[1][b, e], clean[0][b, w], clean[1][b, w], target).sum()
        count += n
    return CrossViewRoPEMetrics(count, *(float(x) for x in (total / count).cpu()))
