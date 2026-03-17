# Data Processing Pipeline

Pre-processes the DROID dataset for GR00T N1.6 training with cached backbone
features and optional 3D token positions (depth-based RoPE).

## Overview

```
Step 0: select_episodes.py           ← CPU, ~1 min
         ↓
Step 1: cache_backbone_features.py   ← GPU array (0-3), ~48 h     ─┐ parallel
Step 2: download_multilab_depths.py  ← CPU single job,  ~12 h     ─┘
         ↓
Step 3: extract_svo_depth.py         ← GPU+Apptainer array (0-31), ~48 h
         ↓
Step 4: build_episode_frame_index.py ← CPU interactive, ~5 min
         ↓
Step 5: check_depth_completeness.py  ← CPU interactive, ~5 min
         ↓
Step 6: cache_depth_features.py      ← CPU array (0-31), ~12 h
```

Steps 1 and 2 can run concurrently — both only depend on Step 0's output.
Step 3 depends on Step 2. Step 6 depends on both Step 1 and Step 3.

---

## Prerequisites

### Environment
```bash
micromamba activate gr00t
```

### Paths
| Variable | Path |
|---|---|
| `DATASET` | `/work/nvme/bgkz/droid_raw_large_superset` |
| `BACKBONE_CACHE` | `/work/nvme/bgkz/droid_multilab_cache` |
| `RAW_DEPTHS` | `/work/nvme/bgkz/droid_multilab_raw` |
| `DEPTH_CACHE` | `/work/nvme/bgkz/droid_multilab_depths` |

### ZED SDK Apptainer image
`/u/hbhatia1/3d-vla/zed_4.0.sif` must exist for Step 3.

ZED calibration files must be present at:
`/u/hbhatia1/.local/share/stereolabs/settings/`

### GCS credentials
Must be configured in `~/.config/gcloud/` (NFS-shared, works on compute nodes).
Verify with:
```bash
gsutil ls gs://gresearch/robotics/droid_raw/1.0.1/ | head -5
```

---

## Step-by-step

### Step 0 — Select episodes
Selects the first 200 episodes (by episode index) from each of 7 labs.

```bash
sbatch scripts/slurm/select_episodes_multilab.slurm
```

**Expected output:** `/work/nvme/bgkz/droid_multilab_depths/selected_episodes.json`
**Expected scale:** 1,400 episodes (200 × 7 labs)
**Runtime:** ~1 min

Verify:
```bash
python3 -c "
import json
s = json.load(open('/work/nvme/bgkz/droid_multilab_depths/selected_episodes.json'))
print('Total:', len(s['episode_indices']))
print('Per lab:', s['per_lab_counts'])
"
```

---

### Step 1 — Cache backbone features  *(can run in parallel with Step 2)*
Runs the frozen GR00T VLM backbone over all selected episodes and saves
embeddings as sharded `.pt` files.

```bash
sbatch --array=0-3 scripts/slurm/cache_features_multilab.slurm
```

**Expected output:** `/work/nvme/bgkz/droid_multilab_cache/`
  - `stats.ready`, `index.pt`, `cache_meta.json`
  - `shard_XXXXX.pt` + `shard_XXXXX.done` (~387 shards)

**Expected scale:** ~396K samples, ~387 shards (shard_size=1024)
**Runtime:** ~48 h (4 A100 GPUs, ~97 shards each)

Resumable: existing `.done` sentinels are skipped automatically.

Verify:
```bash
ls /work/nvme/bgkz/droid_multilab_cache/*.done | wc -l   # should be ~387
cat /work/nvme/bgkz/droid_multilab_cache/cache_meta.json
```

---

### Step 2 — Download raw depth data  *(can run in parallel with Step 1)*
Downloads SVO files, trajectory.h5, and metadata JSONs from GCS for all 1,400
selected episodes.

```bash
sbatch scripts/slurm/download_depths_multilab.slurm
```

**Expected output:** `/work/nvme/bgkz/droid_multilab_raw/{canonical_id}/`
  - `recordings/SVO/*.svo`
  - `trajectory.h5`
  - `metadata_*.json`
  - `.done` sentinel

**Expected scale:** ~31 GB × (1400/1470) ≈ ~30 GB
**Runtime:** ~12 h

Resumable: episodes with `.done` sentinel are skipped.

Verify:
```bash
find /work/nvme/bgkz/droid_multilab_raw -name ".done" | wc -l   # should be ~1400
```

---

### Step 3 — Extract depth frames from SVO files
Decodes ZED SVO recordings to per-episode depth arrays using the ZED SDK
inside an Apptainer container.

```bash
sbatch scripts/slurm/extract_depths_multilab.slurm
```

**Expected output:** `/work/nvme/bgkz/droid_multilab_depths/{canonical_id}/{serial}/`
  - `depth.blosc`   — (T, H, W) float32 compressed depth
  - `shape.npy`     — array shape
  - `intrinsics.npy` — [fx, fy, cx, cy]
  - `.done` sentinel

**Runtime:** ~48 h (32 GPU workers)

Resumable: episodes with `.done` sentinel are skipped.

---

### Step 4 — Build episode frame index  *(interactive, run on login or srun node)*
Builds the mapping from cache global_idx → (canonical_id, frame_idx) and the
serial map (canonical_id → camera serials).

```bash
srun --account=bgkz-delta-cpu --partition=cpu --mem=32G --cpus-per-task=8 \
    --time=00:30:00 --pty bash -l

micromamba activate gr00t
cd /u/hbhatia1/3d-vla
python data_processing/build_episode_frame_index.py \
    --dataset-path           /work/nvme/bgkz/droid_raw_large_superset \
    --raw-dir                /work/nvme/bgkz/droid_multilab_raw \
    --output-dir             /work/nvme/bgkz/droid_multilab_depths \
    --allowed-indices-file   /work/nvme/bgkz/droid_multilab_depths/selected_episodes.json \
    --shard-size             1024 \
    --episode-sampling-rate  0.1
```

**Expected output:**
  - `episode_frame_index.pkl` — list of `{canonical_id, frame_idx}`, length = total samples
  - `serial_map.json` — camera serials per episode

**Runtime:** ~5 min

---

### Step 5 — Check depth completeness  *(interactive)*
Validates that all required depth files exist for each episode in serial_map.

```bash
python data_processing/check_depth_completeness.py \
    --depth-dir  /work/nvme/bgkz/droid_multilab_depths \
    --raw-dir    /work/nvme/bgkz/droid_multilab_raw \
    --serial-map /work/nvme/bgkz/droid_multilab_depths/serial_map.json
```

**Expected output:**
  - `valid_canonical_ids.json`  — list of fully-complete episodes
  - `invalid_episodes.json`     — dict of {canonical_id: [failure reasons]}

**Runtime:** ~5 min

Review failures, then re-run Step 3 for missing episodes if needed.

---

### Step 6 — Cache depth features
Pre-computes per-token 3D world positions (from depth unprojection) and saves
them as depth shard files alongside the backbone cache.

```bash
sbatch scripts/slurm/cache_depth_features_multilab.slurm
```

**Expected output:** `/work/nvme/bgkz/droid_multilab_cache/`
  - `depth_shard_XXXXX.pt` + `depth_shard_XXXXX.done` (~387 shards)

**Expected scale:** ~387 shards, ~13 per rank (32 workers)
**Runtime:** ~12 h

Resumable: shards with `.done` sentinel are skipped.

Verify:
```bash
ls /work/nvme/bgkz/droid_multilab_cache/depth_shard_*.done | wc -l   # should be ~387
```

---

## Validation checklist

```bash
# 1. Backbone shards complete
ls /work/nvme/bgkz/droid_multilab_cache/shard_*.done | wc -l

# 2. Depth shards complete
ls /work/nvme/bgkz/droid_multilab_cache/depth_shard_*.done | wc -l

# 3. Valid episode count
python3 -c "
import json
ids = json.load(open('/work/nvme/bgkz/droid_multilab_depths/valid_canonical_ids.json'))
print('Valid episodes:', len(ids))
"

# 4. Cache metadata
cat /work/nvme/bgkz/droid_multilab_cache/cache_meta.json

# 5. Index sanity
python3 -c "
import torch
idx = torch.load('/work/nvme/bgkz/droid_multilab_cache/index.pt', weights_only=True)
print('Total samples:', len(idx['shard_idx']))
print('Total shards: ', int(idx['shard_idx'].max()) + 1)
"
```

---

## Launching training

After the full pipeline completes:

```bash
python -m gr00t.experiment.launch_finetune \
    --base-model-path     nvidia/GR00T-N1.6-3B \
    --dataset-path        /work/nvme/bgkz/droid_raw_large_superset \
    --embodiment-tag      OXE_DROID \
    --cached-backbone-dir /work/nvme/bgkz/droid_multilab_cache \
    --depth-dir           /work/nvme/bgkz/droid_multilab_depths \
    --shard-size          1024 \
    --episode-sampling-rate 0.1 \
    --allowed-indices-file /work/nvme/bgkz/droid_multilab_depths/selected_episodes.json \
    ... (other training args)
```

---

## Resumability

Every step uses `.done` sentinel files:
- Backbone shards: `{BACKBONE_CACHE}/shard_XXXXX.done`
- Episode downloads: `{RAW_DEPTHS}/{canonical_id}/.done`
- Depth extraction: `{DEPTH_CACHE}/{canonical_id}/{serial}/.done`
- Depth cache shards: `{BACKBONE_CACHE}/depth_shard_XXXXX.done`

Re-submitting any SLURM job will skip already-completed work automatically.

---

## Orchestration script

To submit the full pipeline with automatic SLURM dependencies:

```bash
# Dry run (echo commands, no submission)
bash scripts/reprocess_all.sh --dry-run --all

# Submit full pipeline
bash scripts/reprocess_all.sh --all

# Submit only one step
bash scripts/reprocess_all.sh --step 1
```
