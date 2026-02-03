"""
Steps 3–5: Compute per-axis RoPE frequency norms and aggregate.

- Step 3: L2 norm per axis (x, y, z) on last dimension of [..., D_axis/2, 2].
- Step 4: Concatenate axis norms and axis metadata.
- Step 5: Average over batch, sequence, heads -> [num_layers, D/2].
"""

import torch


def compute_axis_dims(feature_dim: int):
    """
    Match RotaryPositionEncoding3D: D_x, D_y, D_z per axis.
    Returns (D_x, D_y, D_z) such that D_x + D_y + D_z = feature_dim.
    """
    dx = dy = feature_dim // 3
    if dx % 2 == 1:
        dx -= 1
        dy -= 1
    dz = feature_dim - dx - dy
    return dx, dy, dz


# --- Step 3: Compute norms per axis ---


def norms_per_axis_3d(
    queries_x: torch.Tensor,
    queries_y: torch.Tensor,
    queries_z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Step 3: L2 norm on last dimension for each axis.

    Inputs: [B, L, D_axis] per axis (3D RoPE splits full embed_dim, no head dim).
    Reshape to [B, L, D_axis/2, 2], L2 norm on last dim -> [B, L, D_axis/2].
    """
    def norm_axis(q: torch.Tensor) -> torch.Tensor:
        B, L, D = q.shape
        half = D // 2
        q_reshaped = q.view(B, L, half, 2)
        return torch.linalg.norm(q_reshaped, dim=-1)

    norms_x = norm_axis(queries_x)
    norms_y = norm_axis(queries_y)
    norms_z = norm_axis(queries_z)
    return norms_x, norms_y, norms_z


# --- Step 4: Concatenate axis norms and metadata ---


def concat_axis_norms(
    norms_x: torch.Tensor,
    norms_y: torch.Tensor,
    norms_z: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Step 4: Concatenate norms along frequency dimension and build axis metadata.

    Inputs:
        norms_x [B, L, D_x/2], norms_y [B, L, D_y/2], norms_z [B, L, D_z/2]
    Outputs:
        combined_norms [B, L, D/2]
        axis_metadata: dict with keys 'x', 'y', 'z' -> (start, end) frequency indices
    """
    combined_norms = torch.cat([norms_x, norms_y, norms_z], dim=-1)
    d_x2 = norms_x.shape[-1]
    d_y2 = norms_y.shape[-1]
    d_z2 = norms_z.shape[-1]
    axis_metadata = {
        "x": (0, d_x2),
        "y": (d_x2, d_x2 + d_y2),
        "z": (d_x2 + d_y2, d_x2 + d_y2 + d_z2),
    }
    return combined_norms, axis_metadata


# --- Step 5: Aggregation ---


def aggregate_norms(combined_norms: torch.Tensor) -> torch.Tensor:
    """
    Step 5: Average across batch and sequence (positions).

    Input: combined_norms [B, L, D/2]
    Output: [D/2]
    """
    return combined_norms.mean(dim=(0, 1))


def aggregate_norms_per_layer(
    list_combined_norms: list[torch.Tensor],
) -> torch.Tensor:
    """
    Step 5 for multiple layers: aggregate each layer then stack.

    Input: list of combined_norms tensors, each [B, L, D/2]
    Output: [num_layers, D/2]
    """
    return torch.stack([aggregate_norms(c) for c in list_combined_norms], dim=0)


# --- Full pipeline: from per-axis queries to final [num_layers, D/2] ---


def queries_to_axis_norms_single_layer(
    queries_x: torch.Tensor,
    queries_y: torch.Tensor,
    queries_z: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Steps 3 + 4 + 5 for a single layer.

    Inputs: queries_x [B,L,D_x], queries_y [B,L,D_y], queries_z [B,L,D_z]
    (3D RoPE: split on full embed_dim, no head dimension.)
    Outputs:
        aggregated [D/2]
        axis_metadata dict
    """
    norms_x, norms_y, norms_z = norms_per_axis_3d(queries_x, queries_y, queries_z)
    combined, axis_metadata = concat_axis_norms(norms_x, norms_y, norms_z)
    return aggregate_norms(combined), axis_metadata


def queries_to_axis_norms_multi_layer(
    layer_queries: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, dict]:
    """
    Full pipeline for multiple layers.

    Input: list of (queries_x, queries_y, queries_z) per layer
    Outputs:
        aggregated [num_layers, D/2]
        axis_metadata (same for all layers)
    """
    list_combined = []
    axis_metadata = None
    for qx, qy, qz in layer_queries:
        norms_x, norms_y, norms_z = norms_per_axis_3d(qx, qy, qz)
        combined, axis_metadata = concat_axis_norms(norms_x, norms_y, norms_z)
        list_combined.append(combined)
    out = aggregate_norms_per_layer(list_combined)
    return out, axis_metadata


def split_queries_by_axis(queries: torch.Tensor, feature_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split full query tensor along last dimension using D_x, D_y, D_z from
    RotaryPositionEncoding3D (block layout: x, y, z). Use when RoPE uses
    block layout [x_block][y_block][z_block].
    """
    D_x, D_y, D_z = compute_axis_dims(feature_dim)
    qx = queries[..., :D_x]
    qy = queries[..., D_x : D_x + D_y]
    qz = queries[..., D_x + D_y :]
    return qx, qy, qz


def split_queries_by_axis_interleaved(queries: torch.Tensor, feature_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split full query tensor for interleaved RoPE layout (x1,y1,z1, x2,y2,z2, ...).
    Use when RotaryPositionEncoding3D uses interleaved position code.
    """
    D_x, D_y, D_z = compute_axis_dims(feature_dim)
    n_bins = min(D_x, D_y, D_z) // 2
    # Interleaved: dims 6*i, 6*i+1 = x bin i; 6*i+2, 6*i+3 = y; 6*i+4, 6*i+5 = z
    idx_x = torch.cat([torch.arange(6 * i, 6 * i + 2, device=queries.device) for i in range(n_bins)])
    idx_y = torch.cat([torch.arange(6 * i + 2, 6 * i + 4, device=queries.device) for i in range(n_bins)])
    idx_z = torch.cat([torch.arange(6 * i + 4, 6 * i + 6, device=queries.device) for i in range(n_bins)])
    base = 6 * n_bins
    rem_x, rem_y, rem_z = D_x - 2 * n_bins, D_y - 2 * n_bins, D_z - 2 * n_bins
    if rem_x > 0:
        idx_x = torch.cat([idx_x, torch.arange(base, base + rem_x, device=queries.device)])
        base += rem_x
    if rem_y > 0:
        idx_y = torch.cat([idx_y, torch.arange(base, base + rem_y, device=queries.device)])
        base += rem_y
    if rem_z > 0:
        idx_z = torch.cat([idx_z, torch.arange(base, base + rem_z, device=queries.device)])
    qx = queries[..., idx_x]
    qy = queries[..., idx_y]
    qz = queries[..., idx_z]
    return qx, qy, qz
