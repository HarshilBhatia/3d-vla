"""
RoPE 3D frequency norm analysis: capture Q/K before RoPE, compute per-axis norms (Steps 3–5).
Used by analyse_qk.py.
"""

from collections import defaultdict, OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from modeling.utils import multihead_custom_attention as mha_module
from modeling.utils.layers import AttentionLayer
from utils.rope3d_frequency_norms import (
    compute_axis_dims,
    queries_to_axis_norms_single_layer,
    split_queries_by_axis_interleaved,
)
from scripts.rope3d_frequency_visualize import (
    visualize_block_bins_per_layer,
    visualize_rope3d_frequency_norms,)


def _layer_name_to_block(name: str) -> str:
    """Extract block name from layer name, e.g. 'self_attn.attn_layers.0' -> 'self_attn'."""
    if ".attn_layers." in name:
        return name.split(".attn_layers.")[0]
    return name


_current_rope_layer_name = [None]
_rope_query_captures = []


def _attention_layer_pre_hook(name):
    def hook(module, args):
        _current_rope_layer_name[0] = name
    return hook


def _patched_mha_forward(
    query, key, value,
    embed_dim_to_check, num_heads,
    in_proj_weight, in_proj_bias,
    dropout_p, out_proj_weight, out_proj_bias,
    training=True, attn_mask=None, rotary_pe=None,
):
    """Capture Q/K after in_proj, before RoPE; move to CPU to avoid OOM."""
    if rotary_pe is not None and _current_rope_layer_name[0] is not None:
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        _rope_query_captures.append((
            _current_rope_layer_name[0],
            q.detach().cpu(),
            k.detach().cpu(),
        ))
    return mha_module._original_mha_forward(
        query, key, value, embed_dim_to_check, num_heads,
        in_proj_weight, in_proj_bias, dropout_p, out_proj_weight, out_proj_bias,
        training=training, attn_mask=attn_mask, rotary_pe=rotary_pe,
    )


def _register_rope_capture(unwrapped_model, rope_layer_names):
    """Register pre-hooks on RoPE AttentionLayers and patch MHA forward."""
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
    Single pass: capture (Q, K) per layer, move to CPU; compute per-block axis norms.
    Returns (per_block_norms, axis_metadata) or (None, None).
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

    by_layer = defaultdict(list)
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
        layer_results_per_head_q = []
        layer_results_per_head_k = []
        layer_results_per_head_bins_q = []
        layer_results_per_head_bins_k = []
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
            qx, qy, qz = split_queries_by_axis_interleaved(q_flat, feature_dim)
            kx, ky, kz = split_queries_by_axis_interleaved(k_flat, feature_dim)
            agg_q, axis_metadata = queries_to_axis_norms_single_layer(qx, qy, qz)
            agg_k, _ = queries_to_axis_norms_single_layer(kx, ky, kz)
            layer_results_q.append(agg_q)
            layer_results_k.append(agg_k)
            Bq, Sq, Eq = q_flat.shape
            Bk, Sk, Ek = k_flat.shape
            assert Eq == feature_dim and Ek == feature_dim
            q_heads = q_flat.view(Bq, Sq, num_attn_heads, head_dim)
            k_heads = k_flat.view(Bk, Sk, num_attn_heads, head_dim)
            per_head_q = torch.linalg.norm(q_heads, dim=-1).mean(dim=(0, 1))
            per_head_k = torch.linalg.norm(k_heads, dim=-1).mean(dim=(0, 1))
            layer_results_per_head_q.append(per_head_q)
            layer_results_per_head_k.append(per_head_k)
            half = head_dim // 2
            q_head_slice = q_heads[..., : half * 2].view(Bq, Sq, num_attn_heads, half, 2)
            k_head_slice = k_heads[..., : half * 2].view(Bk, Sk, num_attn_heads, half, 2)
            per_head_bins_q = torch.linalg.norm(q_head_slice, dim=-1).mean(dim=(0, 1))  # [num_heads, half]
            per_head_bins_k = torch.linalg.norm(k_head_slice, dim=-1).mean(dim=(0, 1))
            layer_results_per_head_bins_q.append(per_head_bins_q)
            layer_results_per_head_bins_k.append(per_head_bins_k)
        if layer_results_q:
            per_block_norms[block_name] = {
                "q": torch.stack(layer_results_q, dim=0),
                "k": torch.stack(layer_results_k, dim=0),
                "per_head_q": torch.stack(layer_results_per_head_q, dim=0),
                "per_head_k": torch.stack(layer_results_per_head_k, dim=0),
                "per_head_bins_q": torch.stack(layer_results_per_head_bins_q, dim=0),
                "per_head_bins_k": torch.stack(layer_results_per_head_bins_k, dim=0),
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
    Capture Q/K before RoPE, compute per-axis norms, save .pt and PNGs. Rank 0 only.
    """
    device = next(model.parameters()).device

    if dist.get_rank() != 0:
        dist.barrier(device_ids=[device.index] if device.type == "cuda" else [])
        return None, None

    if not getattr(mha_module, "_original_mha_forward", None):
        mha_module._original_mha_forward = mha_module.multi_head_attention_forward
        mha_module.multi_head_attention_forward = _patched_mha_forward

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
                "per_head_bins_q": v["per_head_bins_q"].cpu(),
                "per_head_bins_k": v["per_head_bins_k"].cpu(),
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
        print(f"Saved RoPE 3D frequency norms to {path}")
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
            q_bins = visualize_block_bins_per_layer(
                norms_dict["per_head_bins_q"].cpu().float().numpy(),
                log_dir, block_name, kind="query",
            )
            k_bins = visualize_block_bins_per_layer(
                norms_dict["per_head_bins_k"].cpu().float().numpy(),
                log_dir, block_name, kind="key",
            )
            print(f"  [{block_name}] viz={viz_path}, bins Q/K={q_bins}, {k_bins}")

    if dist.get_world_size() > 1:
        dist.barrier(device_ids=[device.index] if device.type == "cuda" else [])
    return per_block_norms, axis_metadata
