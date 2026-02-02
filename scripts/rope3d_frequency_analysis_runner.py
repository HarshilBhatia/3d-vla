"""
Run RoPE 3D frequency norm analysis: capture query tensors from attention
(before RoPE) and compute per-axis norms (Steps 3–5). Used by analyze_rope3d_frequency_norms.py.
"""

from pathlib import Path
from collections import defaultdict, OrderedDict

import torch
import torch.nn.functional as F

from modeling.utils import multihead_custom_attention as mha_module
from modeling.utils.layers import AttentionLayer
from utils.rope3d_frequency_norms import (
    split_queries_by_axis,
    queries_to_axis_norms_single_layer,
    compute_axis_dims,
)
try:
    from rope3d_frequency_visualize import (
        visualize_rope3d_frequency_norms,
        visualize_per_head_mean_norms_per_layer,
    )
except ImportError:
    from scripts.rope3d_frequency_visualize import (
        visualize_rope3d_frequency_norms,
        visualize_per_head_mean_norms_per_layer,
    )


def _layer_name_to_block(name: str) -> str:
    """Extract block name from layer name, e.g. 'self_attn.attn_layers.0' -> 'self_attn'."""
    if ".attn_layers." in name:
        return name.split(".attn_layers.")[0]
    return name


# Global state for hook: which RoPE layer we're currently in (set by pre_hook on AttentionLayer)
_current_rope_layer_name = [None]  # list so hook can mutate


# Per forward call: list of (layer_name, q_cpu, k_cpu) — moved to CPU immediately to avoid OOM
_rope_query_captures = []


def _attention_layer_pre_hook(name):
    def hook(module, args):
        _current_rope_layer_name[0] = name
    return hook


def _patched_multi_head_attention_forward(
    query, key, value,
    embed_dim_to_check, num_heads,
    in_proj_weight, in_proj_bias,
    dropout_p, out_proj_weight, out_proj_bias,
    training=True, attn_mask=None, rotary_pe=None,
):
    """Capture q and k (after in_proj, before RoPE), move to CPU immediately to avoid GPU OOM."""
    if rotary_pe is not None and _current_rope_layer_name[0] is not None:
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        _rope_query_captures.append((
            _current_rope_layer_name[0],
            q.detach().cpu(),
            k.detach().cpu(),
        ))
    return mha_module._original_mha_forward(
        query, key, value,
        embed_dim_to_check, num_heads,
        in_proj_weight, in_proj_bias,
        dropout_p, out_proj_weight, out_proj_bias,
        training=training, attn_mask=attn_mask, rotary_pe=rotary_pe,
    )


def _register_rope_capture(unwrapped_model, rope_layer_names):
    """Register pre-hooks on RoPE AttentionLayers and patch multi_head_attention_forward."""
    handles = []
    for name, mod in unwrapped_model.named_modules():
        if isinstance(mod, AttentionLayer) and getattr(mod, "rotary_pe", False):
            if name not in rope_layer_names:
                rope_layer_names.append(name)
            h = mod.register_forward_pre_hook(_attention_layer_pre_hook(name))
            handles.append(h)
    return handles


def _collect_queries_and_compute_norms(
    model,
    val_loader,
    train_tester,
    feature_dim,
    num_attn_heads,
    max_batches,
    device,
):
    """
    Single pass: capture (q, k) and move to CPU immediately. Compute per-layer
    axis norms per block for both Q and K on CPU. Returns dict[block_name, {'q': norms, 'k': norms}] and axis_metadata.
    """
    global _rope_query_captures
    _rope_query_captures = []
    rope_layer_names = []

    unwrapped = model.module if hasattr(model, "module") else model
    handles = _register_rope_capture(unwrapped, rope_layer_names)
    if not rope_layer_names:
        for h in handles:
            h.remove()
        return None, None

    num_batches = 0
    for batch in val_loader:
        if max_batches is not None and num_batches >= max_batches:
            break
        _current_rope_layer_name[0] = None
        action, action_mask, rgbs, rgb2d, pcds, instr, prop = train_tester.prepare_batch(
            batch, augment=False
        )
        if getattr(train_tester.args, "pre_tokenize", True):
            from modeling.encoder.text import fetch_tokenizers
            tokenizer = getattr(train_tester, "tokenizer", None)
            if tokenizer is None:
                tokenizer = fetch_tokenizers(train_tester.args.backbone)
            instr = tokenizer(instr).to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = model(
                action, action_mask, rgbs, rgb2d, pcds, instr, prop,
                run_inference=True,
                stopgrad_k=0,
            )
        num_batches += 1

    for h in handles:
        h.remove()

    if not _rope_query_captures:
        return None, None

    # All captured tensors are already on CPU
    by_layer = defaultdict(list)  # name -> [(q_cpu, k_cpu), ...]
    for name, q, k in _rope_query_captures:
        by_layer[name].append((q, k))

    blocks_to_layers = OrderedDict()
    for name in rope_layer_names:
        block = _layer_name_to_block(name)
        if block not in blocks_to_layers:
            blocks_to_layers[block] = []
        blocks_to_layers[block].append(name)

    head_dim = feature_dim // num_attn_heads
    per_block_norms = OrderedDict()
    axis_metadata = None
    for block_name, layer_names in blocks_to_layers.items():
        layer_results_q = []
        layer_results_k = []
        layer_results_per_head_q = []  # [num_heads] mean L2 norm per head, per layer
        layer_results_per_head_k = []
        for name in layer_names:
            qk_list = by_layer.get(name, [])
            if not qk_list:
                continue
            q_list = [x[0] for x in qk_list]
            k_list = [x[1] for x in qk_list]
            q_cat = torch.cat(q_list, dim=1)
            k_cat = torch.cat(k_list, dim=1)
            S, B_total, E = q_cat.shape
            q_flat = q_cat.permute(1, 0, 2).contiguous()
            k_flat = k_cat.permute(1, 0, 2).contiguous()
            qx, qy, qz = split_queries_by_axis(q_flat, feature_dim)
            kx, ky, kz = split_queries_by_axis(k_flat, feature_dim)
            agg_q, axis_metadata = queries_to_axis_norms_single_layer(qx, qy, qz)
            agg_k, _ = queries_to_axis_norms_single_layer(kx, ky, kz)
            layer_results_q.append(agg_q)
            layer_results_k.append(agg_k)
            # Per-head mean norm: use each tensor's own shape for view (q and k can differ per layer)
            Bq, Sq, Eq = q_flat.shape
            Bk, Sk, Ek = k_flat.shape
            assert Eq == feature_dim and Ek == feature_dim
            q_heads = q_flat.view(Bq, Sq, num_attn_heads, head_dim)
            k_heads = k_flat.view(Bk, Sk, num_attn_heads, head_dim)
            per_head_q = torch.linalg.norm(q_heads, dim=-1).mean(dim=(0, 1))
            per_head_k = torch.linalg.norm(k_heads, dim=-1).mean(dim=(0, 1))
            layer_results_per_head_q.append(per_head_q)
            layer_results_per_head_k.append(per_head_k)
        if layer_results_q:
            per_block_norms[block_name] = {
                "q": torch.stack(layer_results_q, dim=0),
                "k": torch.stack(layer_results_k, dim=0),
                "per_head_q": torch.stack(layer_results_per_head_q, dim=0),
                "per_head_k": torch.stack(layer_results_per_head_k, dim=0),
            }

    if not per_block_norms:
        return None, None
    return per_block_norms, axis_metadata


def run_rope_frequency_analysis(
    model,
    val_loader,
    train_tester,
    feature_dim,
    num_attn_heads,
    log_dir,
    max_batches=50,
    save_name="rope3d_frequency_norms.pt",
):
    """
    Capture RoPE query tensors, compute per-axis norms (Steps 3–5), save and return.
    Only run on rank 0; other ranks just barrier.
    """
    import torch.distributed as dist
    device = next(model.parameters()).device

    if dist.get_rank() != 0:
        dist.barrier(device_ids=[device.index] if device.type == "cuda" else [])
        return None, None

    if not getattr(mha_module, "_original_mha_forward", None):
        mha_module._original_mha_forward = mha_module.multi_head_attention_forward
        mha_module.multi_head_attention_forward = _patched_multi_head_attention_forward

    per_block_norms, axis_metadata = _collect_queries_and_compute_norms(
        model,
        val_loader,
        train_tester,
        feature_dim,
        num_attn_heads,
        max_batches=max_batches,
        device=device,
    )

    if per_block_norms is not None and log_dir is not None:
        path = log_dir / save_name
        save_norms = {
            k: {
                "q": v["q"].cpu(),
                "k": v["k"].cpu(),
                "per_head_q": v["per_head_q"].cpu(),
                "per_head_k": v["per_head_k"].cpu(),
            }
            for k, v in per_block_norms.items()
        }
        torch.save(
            {
                "per_block_norms": save_norms,
                "axis_metadata": axis_metadata,
                "feature_dim": feature_dim,
                "D_x2": axis_metadata["x"][1] if axis_metadata else None,
                "D_y2": axis_metadata["y"][1] - axis_metadata["y"][0] if axis_metadata else None,
                "D_z2": axis_metadata["z"][1] - axis_metadata["z"][0] if axis_metadata else None,
            },
            path,
        )
        print(f"Saved RoPE 3D frequency norms (Q & K) to {path}")
        print("Axis metadata (freq_idx -> axis):", axis_metadata)
        D_x, D_y, D_z = compute_axis_dims(feature_dim)
        num_freq = next(iter(per_block_norms.values()))["q"].shape[-1]
        print(f"Per-axis dims: D_x={D_x}, D_y={D_y}, D_z={D_z} -> D/2 = {num_freq}")
        print("Blocks:", list(per_block_norms.keys()))

        # Per-block visualization: Q and K (convert bfloat16 -> float32 for matplotlib/numpy)
        base_viz = Path(save_name).stem
        for block_name, norms_dict in per_block_norms.items():
            viz_name = f"{base_viz}_{block_name}.png"
            viz_path = visualize_rope3d_frequency_norms(
                norms_dict["q"].cpu().float().numpy(),
                axis_metadata,
                log_dir,
                save_name=viz_name,
                block_title=block_name,
                norms_k=norms_dict["k"].cpu().float().numpy(),
            )
            print(f"Saved RoPE 3D frequency visualization (Q & K) [{block_name}] to {viz_path}")

        # One PNG per layer: x=head, y=block, color=mean norm (all blocks in one plot per layer)
        per_block_per_head_q = {
            k: v["per_head_q"].cpu().float().numpy()
            for k, v in per_block_norms.items()
        }
        per_block_per_head_k = {
            k: v["per_head_k"].cpu().float().numpy()
            for k, v in per_block_norms.items()
        }
        visualize_per_head_mean_norms_per_layer(per_block_per_head_q, log_dir, kind="query")
        visualize_per_head_mean_norms_per_layer(per_block_per_head_k, log_dir, kind="key")
        print(f"Saved per-head norms to {log_dir / 'per_head_query_norms'} and {log_dir / 'per_head_key_norms'}")

    if dist.get_world_size() > 1:
        dist.barrier(device_ids=[device.index] if device.type == "cuda" else [])
    return per_block_norms, axis_metadata
