### Download data

```
bash download_droid.slurm
```

### Convert to leRobot


```
python /ocean/projects/cis240058p/hbhatia1/convert_droid_rlds_to_lerobot_v2.py \
  --source-dir /ocean/projects/cis240058p/hbhatia1/data/droid_raw_small \
  --output-dir /ocean/projects/cis240058p/hbhatia1/data/droid_v2_small
```

with left/right filtering

```
python convert_droid_rlds_to_lerobot_v2.py \
  --source-dir /ocean/projects/cis240058p/hbhatia1/data/droid_raw_large \
  --output-dir /ocean/projects/cis240058p/hbhatia1/data/droid_raw_large_superset \
  --cam2base-superset /ocean/projects/cis240058p/hbhatia1/data/droid_annotations/cam2base_extrinsic_superset.json
```

### Cache backbone features

```
python cache_backbone_features.py \
  --dataset-path /ocean/projects/cis240058p/hbhatia1/data/droid_v2_small \
  --output-dir /ocean/projects/cis240058p/hbhatia1/data/droid_v2_small_cache \
  --batch-size 8
```

Use the same dataset path and, if you change them, the same --shard-size (default 1024) and --episode-sampling-rate (default 0.1) that you will use for training. Optional: --resume to skip already-cached samples.

This writes feat_00000000.pt, feat_00000001.pt, ... and cache_meta.json under --output-dir.

### Train using the cache
```
python -m gr00t.experiment.launch_finetune \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path /ocean/projects/cis240058p/hbhatia1/data/droid_v2_small \
  --embodiment-tag OXE_DROID \
  --cached-backbone-dir /ocean/projects/cis240058p/hbhatia1/data/droid_v2_small_cache  \
  --output-dir ./outputs_cached \
  --global-batch-size 64 \
  --max-steps 10000
```