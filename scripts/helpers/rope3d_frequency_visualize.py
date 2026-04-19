"""Visualize RoPE 3D frequency norms: heatmaps and per-axis bar charts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DPI = 150
FIG_SIZE = (12, 6)


def _to_float32(arr):
    """Convert array to float32 (matplotlib does not support bfloat16)."""
    if hasattr(arr, "cpu"):
        return arr.cpu().float().numpy()
    return np.asarray(arr, dtype=np.float32)


def _plot_norms_row(axes_row, norms_per_layer, axis_metadata, title_prefix, num_layers):
    """One row: heatmap (col 0) and per-axis bar chart (col 1)."""
    x_start, x_end = axis_metadata["x"]
    y_start, y_end = axis_metadata["y"]
    z_start, z_end = axis_metadata["z"]

    # Heatmap
    ax = axes_row[0]
    im = ax.imshow(
        norms_per_layer,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
    )
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("Layer")
    ax.set_yticks(np.arange(num_layers))
    ax.set_yticklabels(np.arange(num_layers))
    ax.set_title(title_prefix)
    ax.axvline(x=x_end - 0.5, color="white", linewidth=1, linestyle="--")
    ax.axvline(x=y_end - 0.5, color="white", linewidth=1, linestyle="--")
    ax.set_ylim(num_layers - 0.5, -1.5)
    ax.text((x_start + x_end - 1) / 2, -1.2, "x", ha="center", fontsize=10)
    ax.text((y_start + y_end - 1) / 2, -1.2, "y", ha="center", fontsize=10)
    ax.text((z_start + z_end - 1) / 2, -1.2, "z", ha="center", fontsize=10)
    plt.colorbar(im, ax=ax, label="L2 norm")

    # Per-axis bar chart
    ax = axes_row[1]
    mean_x = norms_per_layer[:, x_start:x_end].mean(axis=1)
    mean_y = norms_per_layer[:, y_start:y_end].mean(axis=1)
    mean_z = norms_per_layer[:, z_start:z_end].mean(axis=1)
    x_layers = np.arange(num_layers)
    width = 0.25
    ax.bar(x_layers - width, mean_x, width, label="x-axis", color="C0", alpha=0.9)
    ax.bar(x_layers, mean_y, width, label="y-axis", color="C1", alpha=0.9)
    ax.bar(x_layers + width, mean_z, width, label="z-axis", color="C2", alpha=0.9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean L2 norm")
    ax.set_title(f"{title_prefix} — per axis by layer")
    ax.legend()
    ax.set_xticks(x_layers)


def visualize_rope3d_frequency_norms(
    norms_per_layer,
    axis_metadata,
    log_dir,
    save_name="rope3d_frequency_norms.png",
    figsize=FIG_SIZE,
    dpi=DPI,
    block_title=None,
    norms_k=None,
):
    """
    Plot RoPE 3D frequency norms for one block (SA or CA). If norms_k is provided,
    plots both Query and Key in a 2x2 layout (row 0: Q, row 1: K).

    norms_per_layer: [num_layers, D/2] (numpy or torch) — query norms
    norms_k: optional [num_layers, D/2] — key norms (same layout as Q)
    axis_metadata: dict with 'x', 'y', 'z' -> (start, end) frequency indices
    log_dir: directory to save the figure
    block_title: optional label (e.g. 'self_attn', 'rotation_self_attn')
    """
    norms_per_layer = _to_float32(norms_per_layer)
    num_layers, num_freq = norms_per_layer.shape
    x_start, x_end = axis_metadata["x"]
    y_start, y_end = axis_metadata["y"]
    z_start, z_end = axis_metadata["z"]
    title_suffix = f" — {block_title}" if block_title else ""

    if norms_k is not None:
        norms_k = _to_float32(norms_k)
        assert norms_k.shape == norms_per_layer.shape
        # 2x2: row 0 = Q (heatmap, bar), row 1 = K (heatmap, bar)
        fig, axes = plt.subplots(2, 2, figsize=(figsize[0], figsize[1] * 2))
        _plot_norms_row(
            axes[0], norms_per_layer, axis_metadata,
            f"Query (Q) RoPE 3D frequency norms{title_suffix}", num_layers,
        )
        _plot_norms_row(
            axes[1], norms_k, axis_metadata,
            f"Key (K) RoPE 3D frequency norms{title_suffix}", num_layers,
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        _plot_norms_row(
            axes, norms_per_layer, axis_metadata,
            f"RoPE 3D frequency norms{title_suffix}", num_layers,
        )

    plt.tight_layout()
    out_path = Path(log_dir) / save_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return str(out_path)


def _block_name_to_filename(block_name: str) -> str:
    """Sanitize block name for filenames."""
    return block_name.replace(".", "_")


def visualize_block_bins_per_layer(
    norms_per_layer_head_bins,
    log_dir,
    block_name,
    kind="query",
    figsize_per_subplot=(5, 3),
    dpi=DPI,
):
    """
    One PNG per block: one subplot per layer. Y = freq bin, X = head.
    norms_per_layer_head_bins: [num_layers, num_heads, head_dim/2].
    Note: per-head bins do not carry 3D axis info (each head uses one axis slice).
    """
    norms = _to_float32(norms_per_layer_head_bins)
    num_layers, num_heads, num_bins = norms.shape
    out_dir = Path(log_dir) / "per_block_bins"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = _block_name_to_filename(block_name) + f"_{kind}.png"
    path = out_dir / fname

    ncols = min(num_layers, 4)
    nrows = (num_layers + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows),
        squeeze=False,
    )
    axes = axes.flatten()
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        im = ax.imshow(
            norms[layer_idx].T,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
        )
        ax.set_xlabel("Head")
        ax.set_ylabel("Freq bin")
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xticks(np.arange(num_heads))
        ax.set_xticklabels(np.arange(num_heads))
        ax.set_yticks(np.linspace(0, num_bins - 1, min(10, num_bins), dtype=int))
        ax.set_yticklabels(np.linspace(0, num_bins - 1, min(10, num_bins), dtype=int))
        plt.colorbar(im, ax=ax, label="Mean L2 norm")
    for idx in range(num_layers, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle(f"Mean {kind.capitalize()} L2 norm (per head, per bin) - {block_name}", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return str(path)


def visualize_per_head_mean_norms_per_layer(
    per_block_per_head_norms,
    log_dir,
    kind="query",
    figsize=(10, 6),
    dpi=DPI,
):
    """
    One PNG per layer: x = head, y = block, color = mean norm.
    Saves to log_dir / per_head_{kind}_norms / layer_0.png, ...
    """
    out_dir = Path(log_dir) / f"per_head_{kind}_norms"
    out_dir.mkdir(parents=True, exist_ok=True)
    max_layers = max(arr.shape[0] for arr in per_block_per_head_norms.values())
    num_heads = next(iter(per_block_per_head_norms.values())).shape[1]
    block_names = list(per_block_per_head_norms.keys())
    saved = []
    for layer_idx in range(max_layers):
        rows_data = []
        rows_labels = []
        for block_name in block_names:
            arr = _to_float32(per_block_per_head_norms[block_name])
            if layer_idx < arr.shape[0]:
                rows_data.append(arr[layer_idx])
                rows_labels.append(block_name)
        if not rows_data:
            continue
        mat = np.array(rows_data)
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_xlabel("Head (head dim)")
        ax.set_ylabel("Block")
        ax.set_yticks(np.arange(len(rows_labels)))
        ax.set_yticklabels(rows_labels, fontsize=8)
        ax.set_xticks(np.arange(num_heads))
        ax.set_xticklabels(np.arange(num_heads))
        ax.set_title(f"Mean {kind.capitalize()} L2 norm — layer {layer_idx}")
        plt.colorbar(im, ax=ax, label="Mean L2 norm")
        plt.tight_layout()
        path = out_dir / f"layer_{layer_idx}.png"
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()
        saved.append(str(path))
    return saved
