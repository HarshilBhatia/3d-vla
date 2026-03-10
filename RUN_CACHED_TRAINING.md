# Running training with cached VLM backbone

This lets you run training **without** the 2B VLM backbone forward: backbone features are precomputed once, then training only runs the DiT action head.

## 1. Cache backbone features (once)

Run on a machine with an Ampere (or compatible) GPU and enough VRAM for the full model. Use the **same** dataset path, `--shard-size`, and `--episode-sampling-rate` you will use for training (defaults: 1024 and 0.1).

```bash
cd Isaac-GR00T

python cache_backbone_features.py \
  --dataset-path /ocean/projects/cis240058p/hbhatia1/data/droid_raw_large_superset \
  --output-dir /ocean/projects/cis240058p/hbhatia1/data/droid_raw_large_superset_cache \
  --batch-size 32
```

Optional:
- `--resume` – skip samples that already have a `feat_*.pt` file (for resuming interrupted runs).
- `--shard-size 1024` – must match training (default 1024).
- `--episode-sampling-rate 0.1` – must match training (default 0.1).

Output:
- `output-dir/feat_00000000.pt`, `feat_00000001.pt`, ... (one per sample).
- `output-dir/cache_meta.json` (dataset path, num_samples, etc.).

## 2. Train using the cache

Point training at the **same** dataset and at the cache directory. The dataloader will load `feat_*.pt` instead of running the backbone.

```bash
cd Isaac-GR00T

python -m gr00t.experiment.launch_finetune \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path /path/to/your/lerobot_dataset \
  --embodiment-tag OXE_DROID \
  --cached-backbone-dir /path/to/cache_output \
  --output-dir ./outputs_cached \
  --global-batch-size 64 \
  --max-steps 10000
```

Important:
- `--dataset-path` must be the same as in step 1.
- `--shard-size` and `--episode-sampling-rate` default to 1024 and 0.1; if you changed them in step 1, pass the same values here.

With `--cached-backbone-dir` set, the backbone is not run during training (only the action head is), so you need much less VRAM and can train without loading the full 2B backbone.

## Without the finetune launcher

If you build the config yourself (e.g. from YAML or code), set:

```python
config.data.cached_backbone_dir = "/path/to/cache_output"
```

and use the same dataset path and data config (shard_size, episode_sampling_rate, seed) as when creating the cache.
