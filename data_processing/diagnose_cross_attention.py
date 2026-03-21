"""Comprehensive cross-attention diagnostic for AlternateVLDiT.

Tests A, B, C across multiple checkpoints at N batches each:
  A) Symmetric ablation: loss% change when image tokens masked vs text tokens masked
  B) Gradient norms: mean ||dL/d(backbone_features)|| at image vs text token positions
  C) Per-block attention entropy: image cross-attn blocks vs text cross-attn blocks
     (higher entropy = more diffuse/uniform attention = block not used selectively)

How the ablations work in AlternateVLDiT:
  - image_mask = zeros  → image blocks see NOTHING, text blocks see ALL tokens  (image ablation)
  - image_mask = backbone_attention_mask → text blocks see NOTHING, image blocks see ALL tokens  (text ablation)

Usage:
    micromamba run -n gr00t python data_processing/diagnose_cross_attention.py \\
        --backbone-cache-dir /work/nvme/bgkz/droid_multilab_cache_ext2 \\
        --depth-cache-dir /work/nvme/bgkz/droid_multilab_depth_cam2cam_ext2 \\
        --n-batches 50 --batch-size 8 \\
        --checkpoints \\
            "groot_base:nvidia/GR00T-N1.6-3B" \\
            "groot_droid:nvidia/GR00T-N1.6-DROID" \\
            "my_baseline:/work/hdd/bgkz/hbhatia1/multilab_baseline_ext2/multilab-baseline-ext2" \\
            "my_3d:/work/hdd/bgkz/hbhatia1/multilab_3d_cam2cam/multilab-3d-cam2cam-ext2"
"""
import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention-capturing processor
# ---------------------------------------------------------------------------

class CapturingCrossAttnProcessor:
    """Cross-attention processor that captures attention weights for entropy analysis.

    Drop-in replacement for RoPE3DCrossAttnProcessor. Stores last_weights after
    each call; extract before the next forward pass overwrites it.
    Computes attention manually (needed to obtain weights) but uses SDPA for the
    actual output for numerical stability.
    """

    def __init__(self):
        self.last_weights: Optional[torch.Tensor] = None  # [B, H, S_q, S_k], CPU

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
        from gr00t.model.gr00t_n1d6.rope_3d import apply_3d_rope_to_keys

        B, seq_len_q, _ = hidden_states.shape
        head_dim = attn.to_k.out_features // attn.heads

        query = attn.to_q(hidden_states)
        kv_src = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(kv_src)
        value = attn.to_v(kv_src)

        # Multi-head reshape: [B, H, S, head_dim]
        query = query.view(B, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key.view(B, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(B, -1, attn.heads, head_dim).transpose(1, 2)

        if rope_cos is not None and rope_sin is not None:
            key = apply_3d_rope_to_keys(key, rope_cos.to(key.dtype), rope_sin.to(key.dtype))
        if rope_cos_q is not None and rope_sin_q is not None:
            query = apply_3d_rope_to_keys(query, rope_cos_q.to(query.dtype), rope_sin_q.to(query.dtype))

        # Build SDPA-compatible bias
        attn_bias: Optional[torch.Tensor] = None
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attn_bias = attention_mask.unsqueeze(1).unsqueeze(1)  # [B,1,1,S_k]
            else:
                attn_bias = attention_mask

        # Manually compute weights for entropy (float32 for stability)
        scale = head_dim ** -0.5
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
        if attn_bias is not None:
            if attn_bias.dtype == torch.bool:
                scores = scores.masked_fill(~attn_bias, float("-inf"))
            else:
                scores = scores + attn_bias.float()
        weights = torch.softmax(scores, dim=-1)  # [B, H, S_q, S_k]
        if not torch.isnan(weights).any():
            self.last_weights = weights.detach().cpu()

        # SDPA for the actual output
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_bias, dropout_p=0.0, is_causal=False
        )
        out = out.transpose(1, 2).reshape(B, -1, attn.heads * head_dim).to(query.dtype)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_entropy(weights: torch.Tensor) -> float:
    """Mean attention entropy (nats) over all (batch, head, query) positions.

    weights: [B, H, S_q, S_k] — must sum to 1 over dim=-1.
    High entropy = diffuse/uniform attention (block not selectively used).
    """
    eps = 1e-9
    h = -(weights * (weights + eps).log()).sum(dim=-1)  # [B, H, S_q]
    return h.mean().item()


def _is_alternate_vl_dit(model) -> bool:
    from gr00t.model.modules.dit import AlternateVLDiT
    return (
        hasattr(model, "action_head")
        and hasattr(model.action_head, "model")
        and isinstance(model.action_head.model, AlternateVLDiT)
    )


def _install_capturing_processors(model) -> Dict[int, object]:
    """Swap in CapturingCrossAttnProcessor on all even (cross-attn) blocks. Return originals."""
    dit = model.action_head.model
    originals = {}
    for idx, block in enumerate(dit.transformer_blocks):
        if idx % 2 == 0:
            originals[idx] = block.attn1.processor
            block.attn1.set_processor(CapturingCrossAttnProcessor())
    return originals


def _restore_processors(model, originals: Dict[int, object]):
    dit = model.action_head.model
    for idx, proc in originals.items():
        dit.transformer_blocks[idx].attn1.set_processor(proc)


def _get_capturing_processors(model) -> List[Tuple[int, CapturingCrossAttnProcessor]]:
    dit = model.action_head.model
    return [
        (idx, block.attn1.processor)
        for idx, block in enumerate(dit.transformer_blocks)
        if idx % 2 == 0
    ]


def _block_role(idx: int, attend_text_every_n_blocks: int) -> str:
    """'text' for text cross-attn blocks, 'image' for image cross-attn blocks."""
    return "text" if idx % (2 * attend_text_every_n_blocks) == 0 else "image"


def _build_inputs(bb, rows, depth_cache, shard_idx, state_dim, device, dtype):
    bam = bb["backbone_attention_mask"][rows]
    if bam.dim() == 3:
        bam = bam.squeeze(-1)
    im = bb["image_mask"][rows]
    if im.dim() == 3:
        im = im.squeeze(-1)

    backbone_features = bb["backbone_features"][rows].to(device, dtype=dtype)
    backbone_attention_mask = bam.to(device)
    image_mask = im.bool().to(device)
    B = backbone_features.shape[0]

    inputs = {
        "backbone_features": backbone_features,
        "backbone_attention_mask": backbone_attention_mask,
        "image_mask": image_mask,
        "state": torch.zeros(B, 1, state_dim, device=device, dtype=dtype),
        "action": torch.zeros(B, 16, state_dim, device=device, dtype=dtype),
        "action_mask": torch.ones(B, 16, state_dim, device=device, dtype=dtype),
        "embodiment_id": torch.full((B,), 16, device=device, dtype=torch.long),
    }
    if depth_cache is not None and shard_idx in depth_cache:
        inputs["token_positions_3d"] = depth_cache[shard_idx][rows].to(device, dtype=dtype)
    return inputs


# ---------------------------------------------------------------------------
# Per-model diagnostics
# ---------------------------------------------------------------------------

def run_diagnostics(model, state_dim, backbone_shards, depth_cache, n_batches, batch_size, device):
    """Run A+B+C diagnostics. Returns result dict or None if not AlternateVLDiT."""
    if not _is_alternate_vl_dit(model):
        print("  WARNING: not AlternateVLDiT — cannot separate image/text tokens")
        return None

    dit = model.action_head.model
    attend_n = getattr(dit, "attend_text_every_n_blocks", 2)

    originals = _install_capturing_processors(model)
    capturing = _get_capturing_processors(model)

    block_entropies: Dict[int, List[float]] = {idx: [] for idx, _ in capturing}
    losses_normal, losses_no_image, losses_no_text = [], [], []
    grad_norms_image, grad_norms_text = [], []

    for i in range(n_batches):
        sf = random.choice(backbone_shards)
        shard_idx = int(sf.stem.split("_")[-1])
        bb = torch.load(sf, weights_only=True, map_location="cpu")
        N = bb["backbone_features"].shape[0]
        rows = random.sample(range(N), min(batch_size, N))

        inputs = _build_inputs(bb, rows, depth_cache, shard_idx, state_dim, device, model.dtype)
        image_mask = inputs["image_mask"]
        backbone_attention_mask = inputs["backbone_attention_mask"]

        # ---- A + C: normal forward, then two ablated forwards ----
        with torch.no_grad():
            out = model(inputs)
            losses_normal.append(out["loss"].item())

        # C: capture entropy from the normal forward before ablated forwards overwrite it
        for idx, proc in capturing:
            if proc.last_weights is not None:
                block_entropies[idx].append(compute_entropy(proc.last_weights))

        with torch.no_grad():
            # Ablate image: image blocks see nothing, text blocks see all tokens
            inp_no_img = dict(inputs)
            inp_no_img["image_mask"] = torch.zeros_like(image_mask)
            losses_no_image.append(model(inp_no_img)["loss"].item())

            # Ablate text: text blocks see nothing, image blocks see all tokens
            inp_no_txt = dict(inputs)
            inp_no_txt["image_mask"] = backbone_attention_mask.bool()
            losses_no_text.append(model(inp_no_txt)["loss"].item())

        # ---- B: gradient norms ----
        bb_feat = inputs["backbone_features"].clone().requires_grad_(True)
        inp_grad = {**inputs, "backbone_features": bb_feat}
        with torch.enable_grad():
            model(inp_grad)["loss"].backward()

        grad = bb_feat.grad.detach()  # [B, S, D]
        img_pos = image_mask & backbone_attention_mask.bool()
        txt_pos = (~image_mask) & backbone_attention_mask.bool()
        if img_pos.any():
            grad_norms_image.append(grad[img_pos].norm(dim=-1).mean().item())
        if txt_pos.any():
            grad_norms_text.append(grad[txt_pos].norm(dim=-1).mean().item())

        if (i + 1) % 10 == 0:
            print(f"    batch {i+1}/{n_batches} | "
                  f"loss={losses_normal[-1]:.4f} | "
                  f"Δimg={100*(losses_no_image[-1]-losses_normal[-1])/losses_normal[-1]:+.1f}% | "
                  f"Δtxt={100*(losses_no_text[-1]-losses_normal[-1])/losses_normal[-1]:+.1f}%")

    _restore_processors(model, originals)

    # Aggregate
    ln = np.array(losses_normal)
    li = np.array(losses_no_image)
    lt = np.array(losses_no_text)
    rel_img = 100 * (li - ln) / ln
    rel_txt = 100 * (lt - ln) / ln

    text_entropies, image_entropies = [], []
    for idx, vals in block_entropies.items():
        if vals:
            entry = (idx, np.mean(vals))
            (text_entropies if _block_role(idx, attend_n) == "text" else image_entropies).append(entry)

    return {
        "n_batches": n_batches,
        "loss_mean": ln.mean(), "loss_std": ln.std(),
        "rel_img_mean": rel_img.mean(), "rel_img_std": rel_img.std(),
        "rel_txt_mean": rel_txt.mean(), "rel_txt_std": rel_txt.std(),
        "grad_img": np.mean(grad_norms_image) if grad_norms_image else float("nan"),
        "grad_txt": np.mean(grad_norms_text)  if grad_norms_text  else float("nan"),
        "text_entropies": sorted(text_entropies),
        "image_entropies": sorted(image_entropies),
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_model_results(name: str, r: Optional[dict]):
    sep = "=" * 70
    print(f"\n{sep}\nMODEL: {name}", end="")
    if r is None:
        print("  →  SKIPPED (not AlternateVLDiT)\n" + sep)
        return
    print(f"  ({r['n_batches']} batches)\n{sep}")

    print(f"\n[A] SYMMETRIC ABLATION (loss % change when block type masked):")
    print(f"  Baseline loss:         {r['loss_mean']:.4f} ± {r['loss_std']:.4f}")
    print(f"  Ablate IMAGE tokens:   {r['rel_img_mean']:+.2f}% ± {r['rel_img_std']:.2f}%")
    print(f"  Ablate TEXT  tokens:   {r['rel_txt_mean']:+.2f}% ± {r['rel_txt_std']:.2f}%")
    if abs(r['rel_img_mean']) > 1e-6:
        ratio = abs(r['rel_txt_mean']) / abs(r['rel_img_mean'])
        print(f"  Text/image impact ratio: {ratio:.1f}x  (>1 = text more important)")

    print(f"\n[B] GRADIENT NORMS  (||dL / d backbone_features|| per token):")
    g_img, g_txt = r["grad_img"], r["grad_txt"]
    print(f"  Image token positions: {g_img:.2e}")
    print(f"  Text  token positions: {g_txt:.2e}")
    if g_img > 1e-12:
        print(f"  Text/image ratio:      {g_txt/g_img:.1f}x  (>1 = text tokens drive learning)")

    print(f"\n[C] ATTENTION ENTROPY per block (nats; higher = more diffuse):")
    print(f"  Text cross-attn blocks (0,4,8,...):")
    for idx, e in r["text_entropies"]:
        print(f"    block {idx:2d}: {e:.4f}")
    print(f"  Image cross-attn blocks (2,6,10,...):")
    for idx, e in r["image_entropies"]:
        print(f"    block {idx:2d}: {e:.4f}")
    if r["text_entropies"] and r["image_entropies"]:
        mean_te = np.mean([e for _, e in r["text_entropies"]])
        mean_ie = np.mean([e for _, e in r["image_entropies"]])
        print(f"  Mean text={mean_te:.4f}  |  mean image={mean_ie:.4f}  |  image/text={mean_ie/mean_te:.2f}x")


def print_summary_table(all_results: dict):
    print(f"\n\n{'='*100}")
    print("SUMMARY TABLE")
    print(f"{'='*100}")
    hdr = (f"{'Model':<22} {'loss':>8} {'Δimg%':>9} {'Δtxt%':>9} "
           f"{'grad_img':>10} {'grad_txt':>10} {'grad_t/i':>9} "
           f"{'ent_img':>8} {'ent_txt':>8} {'ent_i/t':>8}")
    print(hdr)
    print("-" * 100)
    for name, r in all_results.items():
        if r is None:
            print(f"{name:<22}  N/A")
            continue
        me_te = np.mean([e for _, e in r["text_entropies"]]) if r["text_entropies"] else float("nan")
        me_ie = np.mean([e for _, e in r["image_entropies"]]) if r["image_entropies"] else float("nan")
        g_ratio = r["grad_txt"] / (r["grad_img"] + 1e-12)
        e_ratio = me_ie / (me_te + 1e-9)
        print(
            f"{name:<22} {r['loss_mean']:>8.4f} "
            f"{r['rel_img_mean']:>+9.2f} {r['rel_txt_mean']:>+9.2f} "
            f"{r['grad_img']:>10.2e} {r['grad_txt']:>10.2e} {g_ratio:>9.1f}x "
            f"{me_ie:>8.4f} {me_te:>8.4f} {e_ratio:>8.2f}x"
        )
    print(f"\nKey: Δimg%/Δtxt% = loss change when those tokens masked (more positive = more important)")
    print(f"     grad_t/i > 1 = text tokens have higher gradient (text drives learning)")
    print(f"     ent_i/t  > 1 = image blocks more diffuse (not selectively attending)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone-cache-dir", required=True)
    p.add_argument("--depth-cache-dir", default=None)
    p.add_argument("--n-batches", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--checkpoints", nargs="+", required=True, metavar="NAME:PATH",
        help="Checkpoints to evaluate as 'name:path' pairs",
    )
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load backbone shards (shared across models)
    backbone_shards = sorted(Path(args.backbone_cache_dir).glob("shard_?????.pt"))
    if not backbone_shards:
        raise ValueError(f"No backbone shards found in {args.backbone_cache_dir}")
    print(f"Found {len(backbone_shards)} backbone shards")

    # Load depth shards (optional, shared)
    depth_cache = None
    if args.depth_cache_dir:
        depth_files = sorted(Path(args.depth_cache_dir).glob("depth_shard_?????.pt"))
        if depth_files:
            print(f"Loading {len(depth_files)} depth shards ...")
            depth_cache = {}
            for df in depth_files:
                idx = int(df.stem.split("_")[-1])
                d = torch.load(df, weights_only=True, map_location="cpu")
                depth_cache[idx] = d["token_positions_3d"].share_memory_()
            print("Depth shards loaded")

    # Parse checkpoints
    checkpoints = []
    for spec in args.checkpoints:
        if ":" not in spec:
            raise ValueError(f"Checkpoint spec must be 'name:path', got: {spec}")
        name, path = spec.split(":", 1)
        checkpoints.append((name, path))

    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6

    all_results = {}
    for name, ckpt_path in checkpoints:
        print(f"\n{'#'*70}\n# {name}\n# {ckpt_path}\n{'#'*70}")
        try:
            model = Gr00tN1d6.from_pretrained(ckpt_path, skip_backbone=True)
            model.eval().to(device)
            state_dim = getattr(model.config, "max_state_dim", 29)
            print(f"  Loaded. state_dim={state_dim}  AlternateVLDiT={_is_alternate_vl_dit(model)}")

            r = run_diagnostics(model, state_dim, backbone_shards, depth_cache,
                                 args.n_batches, args.batch_size, device)
            all_results[name] = r
            print_model_results(name, r)
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            all_results[name] = None

    print_summary_table(all_results)


if __name__ == "__main__":
    main()
