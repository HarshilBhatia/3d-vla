# Data Loading Pipeline

## Overview

Training has two phases: an offline **caching phase** that runs the backbone once and saves its
outputs, and an online **training phase** that loads those cached outputs and trains only the DiT
action head.

---

## Dataset: `droid_raw_large_superset`

```
/work/nvme/bgkz/droid_raw_large_superset/
    data/
        chunk-000/
            episode_000000.parquet   # per-frame state, action, language
            videos/
                chunk-000/
                    observation.images.exterior_image_1_left/
                        episode_000000.mp4
                    observation.images.wrist_image_left/
                        episode_000000.mp4
    meta/
        episode_index_to_id.json     # episode_idx → {canonical_id, lab, ...}
        episodes.jsonl               # episode lengths
        stats.json                   # action/state mean, std, min, max
```

- 23,700 total episodes, 6.72M frames, 15 FPS
- 13 source labs; RAIL subset = 1,470 episodes
- 2 cameras per episode: exterior (ext1) + wrist, 180×320, AV1 encoded

---

## Step 1: Sharding — `ShardedSingleStepDataset`

Built once at startup (both caching and training). Pure in-memory — nothing is written to disk.

```
ShardedSingleStepDataset(
    dataset_path, seed=42, shard_size=10000,
    episode_sampling_rate=0.1, labs=RAIL
)
```

**What it does:**

1. Filters to RAIL episodes via `allowed_episode_indices` (lab prefix match on canonical_id)
2. Shuffles episode order with `rng = np.random.default_rng(42)`
3. For each episode, shuffles its frame indices and splits into `num_splits = 1/0.1 = 10` sub-sequences — so 10% of each episode's frames are used
4. Assigns sub-sequences to shards greedily (`argmin(shard_lengths)`) to balance shard sizes
5. Produces `sharded_episodes[37]` — a list of shards, each containing `[(ep_idx, [frame_indices]), ...]`

**Result for RAIL:**
- 37 shards, ~10,000 samples each, 360,736 samples total
- Each shard contains frames from many different episodes


**MAJOR DRAWBACK -- NEEDS TO BE FIXED IN THE FUTURE** deterministic — same seed always produces identical assignment. This is critical
for matching backbone cache rows to training samples.

---

## Step 2: `get_shard(idx)` — loading a shard into memory

Called during both caching and training. Iterates over `sharded_episodes[idx]`:

```python
for ep_idx, step_indices in self.sharded_episodes[idx]:
    episode_data = self.episode_loader[ep_idx]   # one disk read per episode
    for step_index in step_indices:
        datapoints.append(self.get_datapoint(episode_data, step_index))
```

**`episode_loader[ep_idx]`** reads:
- `.parquet` file → `pd.DataFrame` of per-frame state + action + language
- `.mp4` videos (ext1 + wrist) via `torchcodec` — decoded on demand per frame

**`get_datapoint(episode_data, step_index)`** extracts:
- Images at `step_index` (and delta offsets for action horizon)
- State, action, language for that timestep
- Wraps into `VLAStepData`, passes through `Gr00tN1d6Processor`
- Returns a dict of tensors ready for the collator

When `cache_dir` is set, each datapoint gets `cache_global_idx = shard_offset + position_in_shard`.

---

## Step 3a (Offline): Backbone Caching — `cache_backbone_features.py`

Run once as a SLURM array job across multiple GPUs. Each GPU processes a static strided subset
of shards: `rank 0 → shards 0, 7, 14, ...`, `rank 1 → shards 1, 8, 15, ...`

**Per shard:**

```python
datapoints = base_dataset.get_shard(shard_idx)   # ~10K datapoints in episode order

for batch in batches_of(datapoints, size=8):
    backbone_features, attention_mask, image_mask = model.backbone(batch)

# pad all batches within shard to same seq_len
shard_data = {
    "backbone_features":        [10000, seq_len, 2048]   # bfloat16
    "backbone_attention_mask":  [10000, seq_len]
    "image_mask":               [10000, seq_len]
}
torch.save(shard_data, "shard_00001.pt")
```

**Index file** (`index.pt`): maps `cache_global_idx → (shard_idx, row)` for O(1) lookup at
training time.

**Output layout:**
```
/droid_rail_cache/
    stats.ready              # sentinel: rank 0 signals stats are done
    index.pt                 # {shard_idx: Tensor(360736,), row: Tensor(360736,)}
    shard_00000.pt           # dict-of-tensors
    shard_00000.done         # sentinel: shard fully written
    ...
    shard_00036.pt
    cache_meta.json
```

---

## Step 3b (Offline): Depth Caching — `cache_depth_features.py`

Run once as a SLURM array job on CPU nodes (no GPU needed). Reads backbone cache shards for
`image_mask`, groups samples by episode within each shard for efficient depth I/O, and writes
`token_positions_3d` per shard.

**Script:** `data_processing/cache_depth_features.py`
**SLURM:** `scripts/slurm/cache_depth_features.slurm`

**Per shard:**

```python
# Load image_mask from backbone shard (needed to locate image token positions)
image_masks = torch.load("shard_00001.pt")["image_mask"]  # [N, seq_len]

# Group all N samples in this shard by episode
by_episode = defaultdict(list)
for global_idx, row in shard_samples:
    entry = episode_frame_index[global_idx]
    by_episode[entry["canonical_id"]].append((row, entry["frame_idx"]))

# One depth.blosc load per episode (not per sample)
for canonical_id, frame_entries in by_episode.items():
    depth_ext1  = load_depth_episode(depth_dir, canonical_id, ext1_serial)   # loaded once
    depth_wrist = load_depth_episode(depth_dir, canonical_id, wrist_serial)  # loaded once
    for row, frame_idx in frame_entries:
        positions[row] = unproject_patches(depth_ext1[frame_idx], ...) + wrist

torch.save({"token_positions_3d": positions}, "depth_shard_00001.pt")
# positions shape: [10000, seq_len, 3], float32
```

**Why episode-grouped I/O matters:** each `depth.blosc` is ~65MB (full episode). Without grouping,
every sample would be a cache miss. By grouping, one load serves all frames of an episode within
the shard.

**Storage:** `seq_len × 3 × float32 × 360K samples ≈ 800MB` total — trivial compared to the
backbone cache (~13GB).

**Output layout (added to backbone cache dir):**
```
/droid_rail_cache/
    depth_shard_00000.pt     # {"token_positions_3d": Tensor[N, seq_len, 3]}
    depth_shard_00000.done   # completion sentinel
    ...
    depth_shard_00036.pt
```

---

## Step 4 (Online): Training — `ShardedMixtureDataset.__iter__`

Wraps one or more `ShardedSingleStepDataset` instances. Handles shard scheduling, prefetching,
and distributed training.

**Shard schedule:** at the start of each epoch, generates a list of `(dataset_idx, shard_idx)`
pairs — shards sampled with weights proportional to dataset size, shuffled.

**Distributed split:** the schedule is divided across `world_size × num_workers` so each worker
gets a unique non-overlapping subset of shards.

**Iteration loop (per worker):**

```
background thread: get_shard(next_shard_idx)
                   → loads raw episode data from disk
                   → returns ~10K datapoints with cache_global_idx

main thread:       shuffle indices within shard
                   yield one sample at a time → DataLoader
```

The background thread prefetches the next shard while the current shard is being consumed by
the GPU, hiding data loading latency.

---

## Step 5 (Online): Collation — `Gr00tN1d6DataCollator.__call__`

The HuggingFace `Trainer` calls this as the DataLoader's `collate_fn`. Receives a list of raw
sample dicts (one per item in the batch).

**At init (once):** if `depth_shard_?????.pt` files exist in `cache_dir`, all are preloaded into
memory (`_depth_shard_cache: dict[shard_idx → Tensor[N, seq_len, 3]]`). At ~22MB per shard and
37 shards, this costs ~800MB of CPU RAM but makes `token_positions_3d` lookup a pure tensor
index — zero I/O at training time.

**Per sample, `_load_feat(cache_global_idx)`:**

1. Load `index.pt` lazily (once, then cached)
2. Look up `(shard_idx, row)` for this `cache_global_idx`
3. Load `shard_XXXXX.pt` into memory (LRU cache of 1 shard, ~675MB each)
4. Slice row → `backbone_features [seq_len, 2048]`, `image_mask [seq_len]`
5. `token_positions_3d [seq_len, 3]` → `_depth_shard_cache[shard_idx][row]` (tensor index, no I/O)

**Output batch:**
```python
{
    "backbone_features":        [B, seq_len, 2048],   # bfloat16
    "backbone_attention_mask":  [B, seq_len],
    "image_mask":               [B, seq_len],
    "token_positions_3d":       [B, seq_len, 3],      # float32, world-frame XYZ
    "action":                   [B, horizon, action_dim],
    "state":                    [B, state_dim],
}
```

---

## Step 6 (Online): Forward Pass — `Gr00tN1d6ActionHead`

```
backbone_features [B, seq_len, 2048]
    → linear projection → [B, seq_len, dit_dim]

noisy_action [B, horizon, action_dim]
    → action tokens via DiT input projection

DiT (32 AlternateVLDiT blocks):
    even blocks (0,2,...): self-attn → cross-attn with text tokens
    odd  blocks (1,3,...): self-attn → cross-attn with image tokens
                           + 3D RoPE on Keys using token_positions_3d

    → predicted noise → diffusion loss vs ground truth noise
```

---

## Summary Table

| Step | Where | When | I/O |
|------|--------|------|-----|
| Shard assignment | `ShardedSingleStepDataset.__init__` | Once at startup | None (in-memory) |
| Raw episode load | `get_shard()` background thread | Every shard, every epoch | Video + parquet per episode |
| Backbone forward | `cache_backbone_features.py` | Once (offline) | Write shard files (~13GB) |
| Depth unproject + cache | `cache_depth_features.py` | Once (offline) | Read depth.blosc, write depth shards (~800MB) |
| Depth shard preload | `Gr00tN1d6DataCollator.__init__` | Once at training start | Read all depth shards (~800MB) into RAM |
| Backbone feature load | `Gr00tN1d6DataCollator._load_feat` | Every batch | Read ~675MB shard (LRU-1) |
| token_positions_3d lookup | `Gr00tN1d6DataCollator._load_feat` | Every sample | None (tensor index into preloaded RAM) |
| DiT forward + loss | `Gr00tN1d6ActionHead.forward` | Every batch | None |
