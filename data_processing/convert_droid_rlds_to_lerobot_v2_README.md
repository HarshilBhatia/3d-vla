# DROID RLDS to LeRobot v2 converter

This document describes the conversion script and how to map between the converted dataset and the original DROID/annotation data.

## What the converter does

The script converts the DROID RLDS dataset (TensorFlow Datasets) into the LeRobot v2 layout:

- **data/chunk-XXX/** – Parquet files per episode (state, action, indices, task_index; no images).
- **videos/chunk-XXX/** – MP4s per camera view (e.g. `observation.images.wrist_image_left`).
- **meta/** – `info.json`, `tasks.jsonl`, `episodes.jsonl`, `stats.json`, `episode_index_to_id.json`, and optionally `conversion_meta.json`.

Episodes are written with consecutive output indices (0, 1, 2, …) so that `episode_000042` refers to the 43rd episode in the converted dataset.

## Options

| Option | Description |
|--------|-------------|
| `--source-dir` | Directory containing the DROID TFDS data (e.g. `droid_raw_small`). |
| `--output-dir` | Where to write the LeRobot v2 dataset. |
| `--max-episodes` | Optional. Cap the number of episodes read from TFDS. |
| `--cam2base-superset` | Optional. Path to `cam2base_extrinsic_superset.json`. If set, **only episodes that have both left and right camera extrinsics** in that file are converted. |
| `--annotations-dir` | Optional. Directory containing `episode_id_to_path.json`. Defaults to the parent of the file given by `--cam2base-superset` when that option is set. |
| `--num-workers` | Number of parallel workers (default 1). Use 30–50 when running on many CPUs to speed up conversion (two-pass: first pass builds task index, second pass processes episodes in parallel). |

When `--cam2base-superset` is used, the converter loads the superset and `episode_id_to_path.json`, resolves each RLDS episode to the canonical episode ID used in the annotations, and keeps only episodes whose canonical ID is in the superset and has at least two camera-serial extrinsic entries (both cameras). Output indices are still consecutive over this filtered set.

## Mapping back to original data

All conversions preserve enough information to go from a converted episode index back to the original DROID/annotation identity.

### Without filtering

- **meta/episode_index_to_id.json** – Maps output index (string key, e.g. `"0"`) to the stored episode id (string) that the RLDS episode provided (e.g. path or canonical id). Use that value to match back to raw data or annotations as needed.

### With `--cam2base-superset` filtering

- **meta/episode_index_to_id.json** – Maps each output index to an object:
  - `canonical_id`: canonical episode ID used in DROID annotation files (e.g. `cam2base_extrinsic_superset.json`, `episode_id_to_path.json`, `droid_language_annotations.json`).
  - `stored_id`: the RLDS-derived id (path or canonical) for that episode.
- **meta/conversion_meta.json** – Records how the conversion was filtered:
  - `filter`: e.g. `"cam2base_superset_both_cameras"`.
  - `cam2base_superset_path`: absolute path to the superset file used.
  - `included_canonical_episode_ids`: list of canonical episode IDs that were included, in output index order.

To go from a converted episode index to annotations:

1. Open `meta/episode_index_to_id.json` and read the entry for that index. If it is an object, use `canonical_id`.
2. Use that canonical ID in the annotation files (e.g. `cam2base_extrinsic_superset.json`, `droid_language_annotations.json`, `camera_serials.json`).

To see which episodes were included in a filtered run, use `meta/conversion_meta.json` → `included_canonical_episode_ids`.

## Example commands

Full conversion (all episodes), single-threaded:

```bash
python convert_droid_rlds_to_lerobot_v2.py \
  --source-dir /path/to/droid_raw_small \
  --output-dir /path/to/droid_v2_small
```

Same with 40 workers (for a 30–50 CPU run):

```bash
python convert_droid_rlds_to_lerobot_v2.py \
  --source-dir /path/to/droid_raw_small \
  --output-dir /path/to/droid_v2_small \
  --num-workers 40
```

Only episodes with both left and right camera extrinsics (from the superset):

```bash
python convert_droid_rlds_to_lerobot_v2.py \
  --source-dir /path/to/droid_raw_small \
  --output-dir /path/to/droid_v2_small \
  --cam2base-superset /path/to/droid_annotations/cam2base_extrinsic_superset.json
```

If `episode_id_to_path.json` is not in the same directory as the superset:

```bash
python convert_droid_rlds_to_lerobot_v2.py \
  --source-dir /path/to/droid_raw_small \
  --output-dir /path/to/droid_v2_small \
  --cam2base-superset /path/to/cam2base_extrinsic_superset.json \
  --annotations-dir /path/to/droid_annotations
```
