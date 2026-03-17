"""
cache_backbone_features.py

Pre-computes and saves GR00T N1.6 VLM backbone embeddings for the DROID dataset.
Run this ONCE on an Ampere GPU. The cached embeddings can then be used to train
only the DiT action head without ever running the 2B backbone again.

Step 1 - Cache features (run as a SLURM array job):
    sbatch --array=0-7 scripts/slurm/cache_features.slurm

    Each rank processes a static stride of shards:
        rank 0 → shards 0, 8, 16, ...
        rank 1 → shards 1, 9, 17, ...
    No locks needed. Rank 0 also writes stats.ready and index.pt.

Step 2 - Train using cache (no backbone forward; same dataset path & shard/seed):
    python -m gr00t.experiment.launch_finetune \\
        --base-model-path nvidia/GR00T-N1.6-3B \\
        --dataset-path /path/to/your/lerobot_dataset \\
        --embodiment-tag OXE_DROID \\
        --cached-backbone-dir /path/to/cache_dir \\
        ...other training args...

Output layout:
    {output_dir}/
        stats.ready              # sentinel written by rank 0 after stats generation
        index.pt                 # {"shard_idx": Tensor(N,), "row": Tensor(N,)}
        shard_{idx:05d}.pt       # dict-of-tensors: backbone_features, backbone_attention_mask, image_mask
        shard_{idx:05d}.done     # sentinel written after successful shard save
        cache_meta.json          # written by rank 0 at the end
"""

import argparse
import json
import time
import torch
from typing import Optional
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


def _atomic_save(obj, tmp_path: Path, final_path: Path, done_path: Path):
    """Save to a .tmp file, rename to final, then touch .done sentinel."""
    torch.save(obj, tmp_path)
    tmp_path.rename(final_path)
    done_path.touch()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="nvidia/GR00T-N1.6-3B")
    parser.add_argument("--embodiment-tag", type=str, default="OXE_DROID")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=10000,
                        help="Shard size for ShardedSingleStepDataset")
    parser.add_argument("--episode-sampling-rate", type=float, default=0.1,
                        help="Episode sampling rate for sharding")
    parser.add_argument("--video-backend", type=str, default="torchcodec")
    parser.add_argument("--rank", type=int, default=0,
                        help="Worker index (SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--world-size", type=int, default=1,
                        help="Total number of workers (SLURM_ARRAY_TASK_COUNT)")
    parser.add_argument("--stats-timeout-min", type=float, default=30.0,
                        help="Minutes to wait for stats.ready before timing out (non-zero ranks)")
    parser.add_argument("--labs", type=str, nargs="+", default=None,
                        help="Only cache episodes from these labs (e.g. --labs TRI AUTOLab). "
                             "Lab names are the prefix of the canonical episode ID (e.g. TRI+...).")
    parser.add_argument("--allowed-indices-file", type=str, default=None,
                        help="Path to a selected_episodes.json produced by select_episodes.py. "
                             "When provided, uses episode_indices from the file as the allowed set, "
                             "bypassing --labs. Cannot be combined with --labs.")
    return parser.parse_args()


def get_allowed_episode_indices(dataset_path: str, labs: list[str]) -> set:
    """Return episode indices whose canonical ID starts with one of the given lab prefixes."""
    id_map_path = Path(dataset_path) / "meta" / "episode_index_to_id.json"
    id_map = json.load(open(id_map_path))
    allowed = set()
    lab_set = set(labs)
    for idx_str, meta in id_map.items():
        lab = meta["canonical_id"].split("+")[0]
        if lab in lab_set:
            allowed.add(int(idx_str))
    return allowed


def wait_for_stats_ready(stats_ready_path: Path, timeout_min: float, rank: int):
    """Non-zero ranks poll until rank 0 signals stats are ready."""
    deadline = time.time() + timeout_min * 60
    print(f"[rank {rank}] Waiting for stats.ready ...")
    while not stats_ready_path.exists():
        if time.time() > deadline:
            raise TimeoutError(
                f"[rank {rank}] Timed out waiting for {stats_ready_path} after {timeout_min} min"
            )
        time.sleep(5)
    print(f"[rank {rank}] stats.ready found, proceeding.")


def build_index(base_dataset) -> dict:
    shard_idxs = []
    rows = []
    for shard_idx in range(len(base_dataset)):
        n = base_dataset.get_shard_length(shard_idx)
        shard_idxs.extend([shard_idx] * n)
        rows.extend(range(n))
    return {
        "shard_idx": torch.tensor(shard_idxs, dtype=torch.int32),
        "row":       torch.tensor(rows,       dtype=torch.int32),
    }


def main():
    args = parse_args()
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_ready_path = output_dir / "stats.ready"

    # ── Modality config and stats ──────────────────────────────────────────────
    print(f"[rank {args.rank}/{args.world_size}] Using embodiment: {args.embodiment_tag}")
    embodiment_tag = EmbodimentTag[args.embodiment_tag]
    modality_config = MODALITY_CONFIGS[args.embodiment_tag.lower()]

    if args.rank == 0:
        print(f"[rank 0] Generating stats for {args.dataset_path}")
        generate_stats(args.dataset_path)
        generate_rel_stats(args.dataset_path, embodiment_tag)
        stats_ready_path.touch()
        print("[rank 0] stats.ready written.")
    else:
        wait_for_stats_ready(stats_ready_path, args.stats_timeout_min, args.rank)

    # ── Load dataset (sharded, step-level) ─────────────────────────────────────
    if args.labs and args.allowed_indices_file:
        raise ValueError(
            "--labs and --allowed-indices-file are mutually exclusive. "
            "Provide at most one episode filter."
        )

    allowed_episode_indices = None
    if args.labs:
        allowed_episode_indices = get_allowed_episode_indices(args.dataset_path, args.labs)
        print(f"[rank {args.rank}] Lab filter {args.labs}: {len(allowed_episode_indices)} episodes")
    elif args.allowed_indices_file:
        indices_path = Path(args.allowed_indices_file)
        if not indices_path.exists():
            raise ValueError(f"--allowed-indices-file not found: {indices_path}")
        with open(indices_path) as _f:
            _sel = json.load(_f)
        allowed_episode_indices = set(_sel["episode_indices"])
        print(f"[rank {args.rank}] allowed-indices-file: {len(allowed_episode_indices)} episodes")

    print(f"[rank {args.rank}] Loading dataset: {args.dataset_path}")
    base_dataset = ShardedSingleStepDataset(
        dataset_path=args.dataset_path,
        embodiment_tag=embodiment_tag,
        modality_configs=modality_config,
        video_backend=args.video_backend,
        shard_size=args.shard_size,
        episode_sampling_rate=args.episode_sampling_rate,
        seed=42,
        allow_padding=False,
        allowed_episode_indices=allowed_episode_indices,
    )
    total_shards = len(base_dataset)
    total_samples = sum(base_dataset.get_shard_length(i) for i in range(total_shards))
    print(f"[rank {args.rank}] Dataset: {total_shards} shards, {total_samples} samples")

    # ── Rank 0 writes index.pt before any shard processing ────────────────────
    index_path = output_dir / "index.pt"
    if args.rank == 0 and not index_path.exists():
        print("[rank 0] Building and saving index.pt ...")
        index = build_index(base_dataset)
        tmp_index = index_path.with_suffix(".tmp")
        torch.save(index, tmp_index)
        tmp_index.rename(index_path)
        print(f"[rank 0] index.pt written ({len(index['shard_idx'])} entries).")

    # ── Load model (fully frozen) ─────────────────────────────────────────────
    print(f"[rank {args.rank}] Loading model: {args.model_path}")
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
    print(f"[rank {args.rank}] Model loaded and frozen.")

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

    # ── Static strided shard assignment (no locks) ────────────────────────────
    my_shards = range(args.rank, total_shards, args.world_size)
    cached, skipped = 0, 0
    print(f"[rank {args.rank}] Processing {len(my_shards)} shards "
          f"(stride {args.world_size}, starting at {args.rank})")

    with ThreadPoolExecutor(max_workers=4) as executor, torch.inference_mode():
        for shard_idx in tqdm(my_shards, desc=f"Shards [rank {args.rank}]"):
            cache_path = output_dir / f"shard_{shard_idx:05d}.pt"
            done_path  = output_dir / f"shard_{shard_idx:05d}.done"

            # Resume: skip if already completed
            if done_path.exists():
                skipped += base_dataset.get_shard_length(shard_idx)
                continue

            datapoints = base_dataset.get_shard(shard_idx)

            # Accumulate per-key lists, then stack into dict-of-tensors
            all_backbone_features       = []
            all_backbone_attention_mask = []
            all_image_mask              = []

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
                all_backbone_features.append(out["backbone_features"].cpu())
                all_backbone_attention_mask.append(out["backbone_attention_mask"].cpu())
                all_image_mask.append(out["image_mask"].cpu())

            # Pad all batches to the global max seq_len across the shard
            max_len = max(t.shape[1] for t in all_backbone_features)
            def pad_feat(t):
                pad = max_len - t.shape[1]
                return torch.nn.functional.pad(t, (0, 0, 0, pad)) if pad > 0 else t
            def pad_mask(t):
                pad = max_len - t.shape[1]
                return torch.nn.functional.pad(t, (0, pad)) if pad > 0 else t

            shard_data = {
                "backbone_features":       torch.cat([pad_feat(t) for t in all_backbone_features], dim=0),
                "backbone_attention_mask": torch.cat([pad_mask(t) for t in all_backbone_attention_mask], dim=0),
                "image_mask":              torch.cat([pad_mask(t) for t in all_image_mask], dim=0),
            }

            tmp_path = cache_path.with_suffix(".tmp")
            executor.submit(_atomic_save, shard_data, tmp_path, cache_path, done_path)
            cached += len(datapoints)

    # ── Save metadata (rank 0 only) ───────────────────────────────────────────
    print(f"\n[rank {args.rank}] Done! Cached: {cached}, Skipped: {skipped}")

    if args.rank == 0:
        meta = {
            "dataset_path": args.dataset_path,
            "model_path": args.model_path,
            "embodiment_tag": args.embodiment_tag,
            "shard_size": args.shard_size,
            "num_shards": total_shards,
            "num_samples": int(total_samples),
            "file_pattern": "shard_{idx:05d}.pt",
            "format": "dict-of-tensors",
        }
        with open(output_dir / "cache_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Metadata saved to: {output_dir / 'cache_meta.json'}")


if __name__ == "__main__":
    main()
