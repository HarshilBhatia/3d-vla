"""
cache_backbone_features.py

Pre-computes and saves GR00T N1.6 VLM backbone embeddings for the DROID dataset.
Run this ONCE on an Ampere GPU. The cached embeddings can then be used to train
only the DiT action head without ever running the 2B backbone again.

Step 1 - Cache features (run once on a GPU with the VLM):
    cd Isaac-GR00T
    python cache_backbone_features.py \\
        --dataset-path /path/to/your/lerobot_dataset \\
        --output-dir /path/to/cache_dir \\
        --batch-size 8

Step 2 - Train using cache (no backbone forward; same dataset path & shard/seed):
    python -m gr00t.experiment.launch_finetune \\
        --base-model-path nvidia/GR00T-N1.6-3B \\
        --dataset-path /path/to/your/lerobot_dataset \\
        --embodiment-tag OXE_DROID \\
        --cached-backbone-dir /path/to/cache_dir \\
        ...other training args...

Saves one file per shard: shard_{idx:05d}.pt  (a list of per-sample dicts)
"""

import argparse
import json
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm

from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6 as GR00TN1d6
from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.stats import generate_rel_stats, generate_stats
from gr00t.configs.data.data_config import DataConfig
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="nvidia/GR00T-N1.6-3B")
    parser.add_argument("--embodiment-tag", type=str, default="OXE_DROID")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-cached shards (safe to resubmit)")
    parser.add_argument("--shard-size", type=int, default=1024,
                        help="Shard size for ShardedSingleStepDataset")
    parser.add_argument("--episode-sampling-rate", type=float, default=0.1,
                        help="Episode sampling rate for sharding")
    parser.add_argument("--video-backend", type=str, default="torchcodec")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Modality config and stats ──────────────────────────────────────────────
    print(f"Using embodiment: {args.embodiment_tag}")
    embodiment_tag = EmbodimentTag[args.embodiment_tag]
    modality_config = MODALITY_CONFIGS[args.embodiment_tag.lower()]
    print(f"Generating stats for {args.dataset_path}")
    generate_stats(args.dataset_path)
    generate_rel_stats(args.dataset_path, embodiment_tag)

    # ── Load dataset (sharded, step-level) ─────────────────────────────────────
    print(f"Loading dataset: {args.dataset_path}")
    base_dataset = ShardedSingleStepDataset(
        dataset_path=args.dataset_path,
        embodiment_tag=embodiment_tag,
        modality_configs=modality_config,
        video_backend=args.video_backend,
        shard_size=args.shard_size,
        episode_sampling_rate=args.episode_sampling_rate,
        seed=42,
        allow_padding=False,
    )
    total_samples = sum(base_dataset.get_shard_length(i) for i in range(len(base_dataset)))
    print(f"Dataset: {len(base_dataset)} shards, {total_samples} samples")

    # ── Load model (fully frozen) ─────────────────────────────────────────────
    print(f"Loading model: {args.model_path}")
    model = GR00TN1d6.from_pretrained(
        args.model_path,
        tune_llm=False,
        tune_visual=False,
        tune_projector=False,
        tune_diffusion_model=False,
    )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    print("Model loaded and frozen.")

    # ── Processor for dataset (same as training) ───────────────────────────────
    data_config = DataConfig()
    stats = {embodiment_tag.value: base_dataset.get_dataset_statistics()}
    processor = Gr00tN1d6Processor(
        modality_configs={embodiment_tag.value: modality_config},
        statistics=stats,
        image_crop_size=data_config.image_crop_size,
        image_target_size=data_config.image_target_size,
        model_name=model.config.model_name,
        model_type=model.config.backbone_model_type,
        transformers_loading_kwargs={"trust_remote_code": True},
    )
    processor.eval()
    base_dataset.processor = processor

    # ── Compile backbone for faster inference ──────────────────────────────────
    model.backbone = torch.compile(model.backbone, mode="default")

    # ── Cache backbone outputs (iterate shards, batch, run backbone) ──────────
    cached, skipped = 0, 0

    with ThreadPoolExecutor(max_workers=4) as executor, torch.inference_mode():
        for shard_idx in tqdm(range(len(base_dataset)), desc="Shards"):
            cache_path = output_dir / f"shard_{shard_idx:05d}.pt"

            if args.resume and cache_path.exists():
                skipped += base_dataset.get_shard_length(shard_idx)
                continue

            datapoints = base_dataset.get_shard(shard_idx)
            shard_feats = []

            for start in range(0, len(datapoints), args.batch_size):
                batch_dps = datapoints[start : start + args.batch_size]
                batch = processor.collator(batch_dps)
                inputs = batch.data["inputs"] if hasattr(batch, "data") else batch["inputs"]
                batch_gpu = {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    backbone_inputs, _, _ = model.prepare_input(batch_gpu)
                    backbone_outputs = model.backbone(backbone_inputs)

                out = getattr(backbone_outputs, "data", backbone_outputs)
                for i in range(len(batch_dps)):
                    sample_feats = {
                        k: v[i].cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in out.items()
                    }
                    shard_feats.append(sample_feats)

            executor.submit(torch.save, shard_feats, cache_path)
            cached += len(shard_feats)

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta = {
        "dataset_path": args.dataset_path,
        "model_path": args.model_path,
        "embodiment_tag": args.embodiment_tag,
        "num_samples": int(total_samples),
        "cached": int(cached),
        "skipped": int(skipped),
        "file_pattern": "shard_{idx:05d}.pt",
        "format": "shard",
    }
    with open(output_dir / "cache_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Cached: {cached}, Skipped: {skipped}")
    print(f"Metadata saved to: {output_dir / 'cache_meta.json'}")


if __name__ == "__main__":
    main()
