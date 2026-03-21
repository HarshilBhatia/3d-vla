"""
Step 2: Ablate image cross-attention to check if the model uses it at all.

Hypothesis: DiT relies on text blocks and barely uses image cross-attention.
If image ablation barely changes loss → 3D RoPE on image blocks is irrelevant.

Pass:  loss increases > 10% when image tokens are ablated
Fail:  loss barely changes → model ignores image cross-attention

Usage:
    micromamba run -n gr00t python data_processing/ablate_image_attention.py \
        --checkpoint /work/hdd/bgkz/hbhatia1/multilab_3d_cam2cam/multilab-3d-cam2cam-ext2 \
        --backbone-cache-dir /work/nvme/bgkz/droid_multilab_cache_ext2 \
        --depth-cache-dir /work/nvme/bgkz/droid_multilab_depth_cam2cam_ext2 \
        --n-batches 8
"""
import argparse
import sys
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint dir")
    p.add_argument("--backbone-cache-dir", required=True)
    p.add_argument("--depth-cache-dir", default=None)
    p.add_argument("--n-batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.factory import DatasetFactory
    from gr00t.configs.base_config import Config
    import json

    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
    from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor

    print(f"Loading checkpoint: {args.checkpoint}")
    model = Gr00tN1d6.from_pretrained(
        args.checkpoint,
        skip_backbone=True,
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")

    # Load processor (saved in processor/ subdir)
    processor_dir = Path(args.checkpoint) / "processor"
    if not processor_dir.exists():
        processor_dir = Path(args.checkpoint)
    processor = Gr00tN1d6Processor.from_pretrained(str(processor_dir))
    processor._collator.cache_dir = Path(args.backbone_cache_dir)

    # Load depth shards if provided
    depth_cache_dir = Path(args.depth_cache_dir) if args.depth_cache_dir else Path(args.backbone_cache_dir)
    depth_files = sorted(depth_cache_dir.glob("depth_shard_?????.pt"))
    if depth_files:
        print(f"Loading {len(depth_files)} depth shards...")
        processor._collator._depth_shard_cache = {}
        processor._collator._eef_shard_cache = {}
        for df in depth_files:
            idx = int(df.stem.split("_")[-1])
            d = torch.load(df, weights_only=True, map_location="cpu")
            processor._collator._depth_shard_cache[idx] = d["token_positions_3d"].share_memory_()
            if "eef_position_3d" in d:
                processor._collator._eef_shard_cache[idx] = d["eef_position_3d"].share_memory_()
        print(f"Depth shards loaded")
    else:
        print("No depth shards found — running without 3D positions")

    # Load directly from backbone + depth cache shards
    import random
    backbone_dir = Path(args.backbone_cache_dir)
    backbone_shards = sorted(backbone_dir.glob("shard_?????.pt"))
    if not backbone_shards:
        raise ValueError(f"No backbone shards in {args.backbone_cache_dir}")
    print(f"Found {len(backbone_shards)} backbone shards")

    # Fake statistics needed by model — load from processor
    stats_file = Path(args.checkpoint) / "processor" / "statistics.json"
    if not stats_file.exists():
        stats_file = Path(args.checkpoint) / "statistics.json"

    losses_normal = []
    losses_no_image = []

    for i in range(args.n_batches):
        # Pick a random shard and random rows from it
        sf = random.choice(backbone_shards)
        shard_idx = int(sf.stem.split("_")[-1])
        bb = torch.load(sf, weights_only=True, map_location="cpu")
        N = bb["backbone_features"].shape[0]
        rows = random.sample(range(N), min(args.batch_size, N))

        backbone_features = bb["backbone_features"][rows].to(device, dtype=model.dtype)
        backbone_attention_mask = bb["backbone_attention_mask"][rows].squeeze(-1) if bb["backbone_attention_mask"].dim() == 3 else bb["backbone_attention_mask"][rows]
        backbone_attention_mask = backbone_attention_mask.to(device)
        image_mask = bb["image_mask"][rows].squeeze(-1) if bb["image_mask"].dim() == 3 else bb["image_mask"][rows]
        image_mask = image_mask.bool().to(device)

        B = backbone_features.shape[0]
        # Determine state/action dim from model config
        state_dim = model.config.max_state_dim if hasattr(model.config, "max_state_dim") else 29
        action_dim = model.config.max_state_dim if hasattr(model.config, "max_state_dim") else 29
        # state: [B, 1, state_dim], action: [B, 16, action_dim]
        state = torch.zeros(B, 1, state_dim, device=device, dtype=model.dtype)
        action = torch.zeros(B, 16, action_dim, device=device, dtype=model.dtype)
        action_mask = torch.ones(B, 16, action_dim, device=device, dtype=model.dtype)
        embodiment_id = torch.full((B,), 16, device=device, dtype=torch.long)

        inputs = {
            "backbone_features": backbone_features,
            "backbone_attention_mask": backbone_attention_mask,
            "image_mask": image_mask,
            "state": state,
            "action": action,
            "action_mask": action_mask,
            "embodiment_id": embodiment_id,
        }

        # Add 3D positions if available
        collator = processor._collator
        if collator._depth_shard_cache is not None and shard_idx in collator._depth_shard_cache:
            pos = collator._depth_shard_cache[shard_idx][rows].to(device, dtype=model.dtype)
            inputs["token_positions_3d"] = pos

        with torch.no_grad():
            # Normal forward
            out_normal = model(inputs)
            loss_normal = out_normal["loss"].item()
            losses_normal.append(loss_normal)

            # Ablated: zero out image mask → image cross-attn blocks see nothing
            inputs_ablated = dict(inputs)
            inputs_ablated["image_mask"] = torch.zeros_like(inputs["image_mask"])
            out_ablated = model(inputs_ablated)
            loss_ablated = out_ablated["loss"].item()
            losses_no_image.append(loss_ablated)

        print(f"  Batch {i+1}: loss_normal={loss_normal:.4f}  loss_no_image={loss_ablated:.4f}  "
              f"delta={loss_ablated - loss_normal:+.4f}  "
              f"rel_increase={100*(loss_ablated-loss_normal)/loss_normal:+.1f}%")

    import numpy as np
    mean_normal = np.mean(losses_normal)
    mean_ablated = np.mean(losses_no_image)
    rel_increase = 100 * (mean_ablated - mean_normal) / mean_normal

    print(f"\n{'='*60}")
    print(f"RESULTS: mean loss_normal={mean_normal:.4f}  mean loss_no_image={mean_ablated:.4f}")
    print(f"Relative increase: {rel_increase:+.1f}%")
    print(f"\nVERDICT")
    print(f"{'='*60}")
    if rel_increase > 10:
        print(f"  PASS (+{rel_increase:.1f}%): model strongly uses image cross-attention")
    elif rel_increase > 2:
        print(f"  MARGINAL (+{rel_increase:.1f}%): model weakly uses image cross-attention")
    else:
        print(f"  FAIL ({rel_increase:+.1f}%): model ignores image cross-attention → RoPE on image blocks has no effect")


if __name__ == "__main__":
    main()
